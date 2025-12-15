# export OSWORLD_DOCKER_IMAGE=osworld-docker-python3:latest
export RAY_TMPDIR=/home/dongyinpeng/mnt/tjy/ray_tmp1214
mkdir -p $RAY_TMPDIR
docker stop $(docker ps -q)
RAY_PORT=2468
RAY_HEAD_IP=127.0.0.1
# export CUDA_VISIBLE_DEVICES=0,5,6,7
ray start --head --port=$RAY_PORT --num-gpus=8 --resources='{"docker:'$RAY_HEAD_IP'": 128}'
# bash ./examples/osworld_subset32.sh
bash ./examples/my_test.sh
ray stop