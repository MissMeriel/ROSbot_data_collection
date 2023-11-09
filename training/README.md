# Training a DAVE2 model



## Training on slurm
1. Log into the department portal nodes: ``ssh computing-id@portal.cs.virginia.edu``
2. Clone this repo (or your fork of this repo) into your home directory: ``git clone git@github.com:MissMeriel/ROSbot_data_collection.git``
3. Navigate to the training directory: ``cd ~/ROSbot_data_collection/training``
4. Create a python virtual environment and install requirements using the script provided: ``./install.sh``
5. Create a dataset directory and copy your datasets to that directory: 
```
mkdir -p ~/ROSbot_data_collection/datasets
scp -r username@remote:/path/to/dataset ~/ROSbot_data_collection/datasets
```
6. Edit the ``train.sh`` script to point to the dataset parent directory of the dataset you want to train on.
7. Check what slurm gpu nodes are available via `sinfo`. You should see output similar to:
```
computing-id@portal0X:/p/sdbb$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
main*        up 4-00:00:00      2    mix lynx08,puma01
main*        up 4-00:00:00      1  alloc hydro
main*        up 4-00:00:00     28   idle affogato[01-10],cortado[01-10],lynx09,optane01,panther01,slurm[1-5]
gpu          up 4-00:00:00     19    mix adriatic[01-04],cheetah[01,04],jaguar[01-06],lotus,lynx[05-06,11],ristretto01,sds[01-02]
gpu          up 4-00:00:00     16   idle adriatic[05-06],affogato[11-15],cheetah[02-03],lynx[01-04,07,10,12]
nolim        up 20-00:00:0      5   resv doppio[01-05]
nolim        up 20-00:00:0      2   idle epona,heartpiece
gnolim       up 20-00:00:0      3    mix ai[03,06,09]
gnolim       up 20-00:00:0     14   idle ai[01-02,04-05,07-08,10],jinx[01-02],titanx[01-05]
```
Nodes marked `idle` mean they are available for you to launch jobs on them. Refer to the CS documentation here for more info: [CS computing info](https://www.cs.virginia.edu/wiki/doku.php?id=start).
The CS grad student orientation to slurm presentation is helpful as well to get started with slurm: [link to slides](https://www.cs.virginia.edu/wiki/lib/exe/fetch.php?media=introtoslurm.pdf).

7. Launch the job on slurm using one of the following configurations: 
```
sbatch -w ai01 -p gnolim --gres=gpu:1 --exclusive=user train.sh # for gnolim partition nodes
sbatch -w adriatic05 -p gpu --gres=gpu:1 --exclusive=user train.sh # for gpu partition nodes
```
8. Check the job periodically to be sure it is progressing using the `squeue -u $USER` command, and check the log according to the `$SLURM_JOB_ID` in `slurm-$SLURM_JOB_ID.out`.