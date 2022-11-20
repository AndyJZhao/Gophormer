from time import time

import dgl
import torch as th
from torch.utils.data import Subset as thSubset

from Gophormer.proj.data.g_data_loader import SampleGraphSubgraphLoader
from Gophormer.proj.eval import eval_classification
from Gophormer.proj.data import proximity_encoding
from Gophormer.functions import print_log, time2str
from Gophormer.modules.early_stopper import EarlyStopping
from Gophormer.modules.lr_scheduler import get_scheduler
from tqdm import tqdm


## original

def consis_loss(logits, temp):
    # CR loss modified from DGL implementation of GRAND
    ps = th.softmax(logits, dim=1)  # [bsz * n_samples,n_class]
    ps = ps.view(ps.shape[0], ps.shape[1], -1)  # [bsz, n_class , n_samples]

    avg_p = th.mean(ps, dim=2)
    sharp_p = (th.pow(avg_p, 1. / temp) / th.sum(th.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = th.mean(th.sum(th.pow(ps - sharp_p, 2), dim=1, keepdim=True))
    return loss


class SampleGraphTrainer():
    def __init__(self, model, g, cf, features, sup, loss_func=th.nn.CrossEntropyLoss()):
        self.cf = cf
        self.__dict__.update(cf.__dict__)
        self.__dict__.update(sup.__dict__)
        self.model = model
        self.optimizer = th.optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)

        self.stopper = EarlyStopping(patience=cf.stopper_patience, path=cf.checkpoint_file) if cf.early_stop else None
        self.loss_func = loss_func
        self.train_x, self.val_x, self.test_x = \
            [_.to(cf.sample_device) for _ in [sup.train_x, sup.val_x, sup.test_x]]
        self.labels = sup.labels.to(cf.compute_dev)
        self.logger = cf.logger
        self.log = self.logger.log

        # ! Process graph
        g_sample = g.to(cf.compute_dev)
        if self.rm_self_loop:
            g_sample = dgl.remove_self_loop(g_sample)
        if cf.order_bias:
            g_compute = proximity_encoding(g_sample, cf.K, cf.include_identity, cf.rm_self_loop, cf.global_nodes, cf.pe_file)
        else:  # No need for subgraphs
            g_compute = g_sample.to(cf.compute_dev)

        g_compute.ndata['F'] = features.to(g_compute.device)
        g_compute = g_compute.to(cf.compute_dev)
        self.g = g_compute
        cf.max_degree = g_sample.in_degrees().max().item()
        g_compute.ndata['centrality'] = g_sample.in_degrees()

        # ! Get dataloader
        train_data = SampleGraphSubgraphLoader(g_sample, g_compute, cf, is_full=False)
        val_data = SampleGraphSubgraphLoader(g_sample, g_compute, cf, is_full=cf.val_full_inf, n_samples=cf.val_n_samples)

        def _get_loader(dataset, batch_size):
            return th.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,  # small batch for full inference
                collate_fn=collate_fn(),  # * How to form batch
                shuffle=True,
                num_workers=cf.n_workers,
            )

        self.train_loader = _get_loader(thSubset(train_data, self.train_x), cf.batch_size)
        self.val_loader = _get_loader(thSubset(val_data, self.val_x), cf.val_bsz)
        self.test_loader = _get_loader(thSubset(val_data, self.test_x), cf.val_bsz)
        self.tqdm = lambda _, desc: tqdm(_, desc=desc, miniters=20) if cf.tqdm_on else _

        # ! Get LR-Scheduler
        self.scheduler = get_scheduler(cf, self.optimizer, len(self.train_loader))

    def run(self):
        self.log('Start training...')
        training_start_time = time()
        for epoch in range(self.epochs):
            t0 = time()
            cla_loss, cr_loss, loss, train_acc = self._train()
            is_eval = epoch >= self.min_eval_epochs and epoch % self.eval_freq == 0
            val_acc = self._evaluate()[0] if is_eval else 0
            self.logger.dynamic_log({'Epoch': epoch, 'Time': time2str(time() - t0), 'cla_loss': cla_loss, 'cr_loss': cr_loss, 'loss': loss, 'TrainAcc': train_acc, 'ValAcc': val_acc}, log_verbose=1)
            if self.stopper is not None and is_eval:
                if self.stopper.step(val_acc, self.model, epoch):
                    self.log(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break

            if self.scheduler != None:
                if self.scheduler_type == 'DecayAtPlateau':
                    self.scheduler.step(metrics=val_acc)
                self.logger.dynamic_log({'lr': self.optimizer.param_groups[0]["lr"]})

        if self.stopper.best_epoch is not None:
            self.model.load_state_dict(th.load(self.stopper.path))
        else:
            self.stopper.best_epoch = epoch
        self.cf.training_time = time2str(time() - training_start_time)
        self.cf.loss = f'L={loss:.4f}, L_cla={cla_loss:.4f}, L_cr={cr_loss:.4f}'
        return self.model

    def eval_and_save(self):
        val_acc, test_acc = self._evaluate(final_eval=True)

        best_epoch = self.stopper.best_epoch if self.stopper is not None else -1
        result = {'val_acc': round(val_acc, 4), 'test_acc': round(test_acc, 4)}
        res = {"res_file": self.cf.res_file, 'best_epoch': best_epoch, **result}
        self.logger.dynamic_log(result)
        self.logger.log(f'\nTrain seed{self.cf.seed} finished\nResults: {res}\n{self.cf}')
        self.logger.static_log(self.cf.model_conf)
        self.logger.static_log(res)
        self.logger.save()

    def process_batch(self, loader, batch_data, rm_prev=False):
        ret = loader.dataset.dataset.get_batch_data(batch_data, rm_prev=rm_prev)
        return ret

    def _train(self):
        self.model.train()
        loss_list, cla_loss_list, cr_loss_list, batch_time_list = [], [], [], []
        pred = th.ones_like(self.labels).to(self.compute_dev) * -1
        for batch_id, batch_data in enumerate(self.tqdm(self.train_loader, 'Train')):
            # ! Prepare batch data
            node_ids, node_pos, batched_graphs, graph_offset = self.process_batch(self.train_loader, batch_data)
            output_labels = self.labels[node_ids]
            assert th.sum(self.g.ndata['F'][node_ids] != batched_graphs.ndata['F'][node_pos]) == 0  # Center node feat-lookup
            # ! Train Model
            logits = self.model(batched_graphs, node_pos, graph_offset)
            cla_loss = self.loss_func(logits, output_labels) / self.n_samples
            if self.cr_weight > 0:
                cr_loss = consis_loss(logits, self.cr_temperature)
                loss = cla_loss + self.cr_weight * cr_loss
            else:
                cr_loss = th.tensor(-1)
                loss = cla_loss

            pred[node_ids] = th.argmax(logits, dim=1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler != None and self.scheduler_type == 'Cos':
                self.scheduler.step()
                # self.log(f'lr = {self.optimizer.param_groups[0]["lr"]}')
            #
            if batch_id + 1 < len(self.train_loader) or len(self.train_loader) == 1:
                # Metrics of the last batch (high variance) shouldn't be added
                loss_list.append(loss.item())
                cla_loss_list.append(cla_loss.item())
                cr_loss_list.append(cr_loss.item())

        train_acc, train_f1, train_mif1 = eval_classification(pred[self.train_x], self.labels[self.train_x], n_class=self.n_class)
        mean = lambda x: sum(x) / len(x)
        return mean(cla_loss_list), mean(cr_loss_list), mean(loss_list), train_acc

    @th.no_grad()
    def _evaluate(self, final_eval=False):
        def _eval_model(loader, val_x, val_y):
            pred = th.ones_like(self.labels).to(self.compute_dev) * -1
            for batch_id, batch_data in enumerate(self.tqdm(loader, 'Evaluate')):
                node_ids, node_pos, batched_graphs, graph_offset = self.process_batch(loader, batch_data)
                try:
                    _logits = self.model(batched_graphs, node_pos, graph_offset)
                except KeyboardInterrupt:
                    continue
                except:
                    print(f'Batch {node_ids[0]} failed', '')
                    node_ids, node_pos, batched_graphs, graph_offset = self.process_batch(loader, batch_data, rm_prev=True)
                    _logits = self.model(batched_graphs, node_pos, graph_offset)
                # Readout
                ids = [_ + th.arange(0, int(node_ids.shape[0] / self.val_n_samples), device=_logits.device) * self.val_n_samples
                       for _ in range(self.val_n_samples)]
                logits = th.stack([_logits[id] for id in ids]).mean(dim=0)
                pred[node_ids[ids[0]]] = th.argmax(logits, dim=1)

            acc, val_f1, val_mif1 = eval_classification(pred[val_x], val_y, n_class=self.n_class)
            return acc

        self.model.eval()
        val_acc = _eval_model(self.val_loader, self.val_x, self.labels[self.val_x])
        test_acc = _eval_model(self.test_loader, self.test_x, self.labels[self.test_x]) if final_eval else 0
        return val_acc, test_acc


def collate_fn():
    def batcher_dev(batch):
        return th.stack(batch)

    return batcher_dev

#
# def collate_fn():
#     def batcher_dev(batch):
#         '''
#
#         Args:
#             batch: seed node indexes, the local positions relative to center node (indexed by order), dgl batched graphs
#
#         Returns:
#
#         '''
#
#         _node_ids, _graph_list = zip(*batch)
#         # bsz, bsz * n_samples
#         n_samples = len(_graph_list[0])
#         # ! Unpack items
#         node_ids = [id for id in _node_ids for _ in range(n_samples)]  # Repeat node ids to [bsz * n_samples]
#         graph_list = [g[_] for g in _graph_list for _ in range(n_samples)]
#         # local_positions = [pos[_] for pos in _local_positions for _ in range(n_samples)]
#
#         batched_graphs = dgl.batch(graph_list)
#         g_sizes = batched_graphs.batch_num_nodes()
#         graph_offset = th.zeros_like(g_sizes)
#         if len(g_sizes) > 1:
#             graph_offset[range(1, len(g_sizes))] = th.stack([g_sizes[:_].sum() for _ in range(1, len(g_sizes))]).squeeze()
#         # node position at batched graphs = subgraph_prefix + seed_node_local_pos
#         # node_batched_positions = {
#         #     _: [node_pos[_] + graph_offset[graph_id] if node_pos[_] is not None else None
#         #         for graph_id, node_pos in enumerate(local_positions)]
#         #     for _ in range(len(local_positions[0]))}
#         # Center nodes can be batched since they are of the same dimension
#         # node_batched_positions[0] = th.stack(node_batched_positions[0])
#         return th.stack(node_ids), graph_offset, batched_graphs
#         # return th.stack(node_ids), (graph_offset, node_batched_positions), batched_graphs
#
#     return batcher_dev
