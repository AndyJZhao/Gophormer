DIR=$(dirname "${BASH_SOURCE[0]}")
source "${DIR}/shell_env.sh"

screen_name="${1}"
screen -mdS ${screen_name}
screen -S ${screen_name} -X stuff "${SHELL_INIT}\r"
echo "Screen ${screen_name} created and initialized"
screen -r ${screen_name}
