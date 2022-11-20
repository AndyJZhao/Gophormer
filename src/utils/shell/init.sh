# Shared commands for all servers
# <<<<<<<<<<<<<<<<<<<< Project Specific Configs >>>>>>>>>>>>>>>>>>>>>> #
SRC_FOLDER=$(dirname $(dirname $(dirname "${BASH_SOURCE[0]}")))
echo $2
echo "The SRC_FOLDER is $SRC_FOLDER"
python3 ${SRC_FOLDER}/tune/settings/gen_shell_env.py -s "$1" -m "$2"
source "${SRC_FOLDER}/utils/shell/shell_env.sh"
# <<<<<<<<<<<<<<<<<<<< Project Shared Configs >>>>>>>>>>>>>>>>>>>>>> #

export LPM="$LP/src/$PROJ_NAME/models/$MODEL"        # Local Model SRC_FOLDER
export TU="$LP/src/utils/exp/tuner.py"               # Tuner
export SU="$LP/src/utils/exp/summarizer.py -m$MODEL" # Summarizer
export TR="$LPM/train.py"                            # Trainer
export CF="$LPM/config.py"                           # Config

# <<<<<<<<<<<<<<<<<<<< Commands >>>>>>>>>>>>>>>>>>>>>> #
# General
alias gg="watch -n 0.5 'nvidia-smi | ${HTOP_FILE}'"
alias nn="nvidia-smi"
alias force_pull="git fetch --all; git reset --hard HEAD; git merge @{u}"
alias pyt="python"
alias sz='du -lh --max-depth=1'
alias so="${SHELL_INIT}"

# Screen
alias ss='bash $LP/src/utils/shell/create_screen.sh'
alias sr='screen -d -r'
alias sl='screen -ls'

# Proj
alias prt='python $TR'
alias run="python $LP/src/utils/exp/runner.py"
alias sw="python $LP/src/utils/exp/sw_runner.py"
alias check='run -c'
alias sum='python $SU'
alias po='cd $LP; git pull && so' # Pull and update settings
alias psg='source $LP/src/utils/shell/pull_and_summarize.sh'
#alias psg_rui='po_rui;rm -rf results;sum;git add results exp && git commit -m "Results updated at $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)" &&git push --branch rui'
alias add_log='git add log && git commit -m log_added && git push'
alias add_exp_db='git add exp_db && git commit -m exp_db_added && git push'
alias pull_code='rsync -a $RP/src $LP/src && echo Code pushed!'
alias tu='python $TU'
#alias tu='CUDA_VISIBLE_DEVICES=0 python $TU -g0'
#alias tu0='CUDA_VISIBLE_DEVICES=0 python $TU -g0'
#alias tu1='CUDA_VISIBLE_DEVICES=1 python $TU -g1'
#alias tu2='CUDA_VISIBLE_DEVICES=2 python $TU -g2'
#alias tu3='CUDA_VISIBLE_DEVICES=3 python $TU -g3'
#alias tu4='CUDA_VISIBLE_DEVICES=4 python $TU -g4'
#alias tu5='CUDA_VISIBLE_DEVICES=5 python $TU -g5'
#alias tu6='CUDA_VISIBLE_DEVICES=6 python $TU -g6'
#alias tu7='CUDA_VISIBLE_DEVICES=7 python $TU -g7'
#alias tu8='CUDA_VISIBLE_DEVICES=8 python $TU -g8'
#alias tu9='CUDA_VISIBLE_DEVICES=9 python $TU -g9'
#alias tu10='CUDA_VISIBLE_DEVICES=10 python $TU -g10'
#alias tu11='CUDA_VISIBLE_DEVICES=11 python $TU -g11'
#alias tu12='CUDA_VISIBLE_DEVICES=12 python $TU -g12'
#alias tu13='CUDA_VISIBLE_DEVICES=13 python $TU -g13'
#alias tu14='CUDA_VISIBLE_DEVICES=14 python $TU -g14'
#alias tu15='CUDA_VISIBLE_DEVICES=15 python $TU -g15'
#alias tu99='CUDA_VISIBLE_DEVICES=15 python $TU -g-1'

# alias push_results='sum;rsync -a $LP/results $LP/log $RP && echo Results pushed!';

echo $MODEL Configs Updated!
cd $LP
