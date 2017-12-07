set -e

STEERING_VALUES=$1
MAX_SPEED=$2
TIMEOUT=$3
PORT=$4
cp train.py train_tmp.py
sed -i "s/REPLACE_STEERING/${STEERING_VALUES}/g" train_tmp.py
sed -i "s/REPLACE_MAX_SPEED/${MAX_SPEED}/g" train_tmp.py

# start server, client should connect.
( torcs -r  ~/ci/torcs-client/tmp.quickrace.xml > server.out 2>&1 ) & server_pid=$!
echo "$server_pid"
sleep 1

# start client with parameters in subprocess
( python3 train_tmp.py -p $PORT > client.out 2>&1 ) & client_pid=$!
echo "$client_pid"

sleep $TIMEOUT && kill -2 $client_pid
kill -2 $server_pid
return 0
