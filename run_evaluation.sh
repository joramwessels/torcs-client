set -e

STEERING_VALUES=$1
MAX_SPEED=$2
TIMEOUT=$3
cp train.py train_tmp.py
sed -i "s/REPLACE_STEERING/${STEERING_VALUES}/g" train_tmp.py
sed -i "s/REPLACE_MAX_SPEED/${MAX_SPEED}/g" train_tmp.py
# start client with parameters in subprocess
python3 train_tmp.py > client.out 2>&1 & client_pid=$!
echo $client_pid > client.out
# start server, client should connect.
( torcs -r  ~/ci/torcs-client/tmp.quickrace.xml > server.out 2>&1 & ) & pid=$!
( sleep $TIMEOUT && kill -HUP $pid ) 2>/dev/null & watcher=$!
if wait $pid 2>/dev/null; then
    pkill -HUP -P $watcher
    wait $watcher
else
    echo "Timeout\n" > server.out
fi
kill -HUP $client_pid
