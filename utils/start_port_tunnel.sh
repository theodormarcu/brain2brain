# Start Port Tunnelling

ipnport=$1
echo "Port: " $ipnport

ssh -N -f -L localhost:$ipnport:localhost:$ipnport tmarcu@tigergpu.princeton.edu