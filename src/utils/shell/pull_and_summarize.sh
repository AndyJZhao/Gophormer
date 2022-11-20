sum_conf="-m${1}"
if [ "${1}" = "" ]; then
  sum_conf=""
fi
po
rm -rf results/${1}
echo "Sum config is: ${sum_conf}"
sum ${sum_conf}
git add results exp && git commit -m "Results updated at $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)" && git push
