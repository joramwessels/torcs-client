set -e

STEERING_VALUES=$1
MAX_SPEED=$2
TIMEOUT=$3
cp train.py train_tmp.py
sed -i "s/REPLACE_STEERING/${STEERING_VALUES}/g" train_tmp.py
sed -i "s/REPLACE_MAX_SPEED/${MAX_SPEED}/g" train_tmp.py
# start client with parameters in subprocess
( python3 train_tmp.py -p $PORT > client.out 2>&1 ) & client_pid=$!
echo "$client_pid"

sleep $TIMEOUT && kill -9 $server_pid > /dev/null 2>&1
kill -9 $client_pid > /dev/null 2>&1
kill -9 $(lsof -i:$PORT -t) > /dev/null 2>&1
