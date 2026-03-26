# run_exp_deccon_backup.ps1
# Launch a DECOOD-vMF gossip training experiment from PowerShell

$ErrorActionPreference = "Stop"

# ----------------- Resolve repo root -----------------
# If this script is in DFedDG2/scripts/, repo root becomes DFedDG2
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

# ----------------- Config -----------------
$dataset = "office"          # "domainnet", "digit"
$feature_iid = 1
$label_iid = 0
$alpha = 1
$lr = 0.1                    # 0.3 for digits
$decood_loss_code = "EC"
$num_trials = 2
$num_clients = 4             # 4 for office, 5 for domainnet, 5 for digit
$n_rounds = 4
$feat_dim = 512
$local_epochs = 1
$local_bs = 32
$tau = 0.1
$weighted_adj_mat = 1
$use_imb_loss = 0
$backbone = "mobilenet_proj"   # resnet18_proj / resnet34_proj

$topo = "dynamic"              #ring, fc
$dist = "dir"

$data_dir = "../../data/"
$save_folder_name = "results/"
$params_dir = "params/"

$algorithm = "DecoodVMF"
$comm = "comm_vmf_gossip"

# ----------------- Setup -----------------
New-Item -ItemType Directory -Force -Path "logs/experiments" | Out-Null
New-Item -ItemType Directory -Force -Path $save_folder_name | Out-Null
New-Item -ItemType Directory -Force -Path $params_dir | Out-Null

Write-Host "Run Deccon.................................................."

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log_file = "logs/experiments/{0}_{1}_{2}_fi{3}_li{4}_topo{5}_dist_{6}_clients{7}_{8}.log" -f `
    $algorithm, $comm, $dataset, $feature_iid, $label_iid, $topo, $dist, $num_clients, $timestamp

# ----------------- Run -----------------
$pythonArgs = @(
    "run_trainer_dfeddg2.py"
    "--algorithm", $algorithm
    "--comm", $comm
    "--topo", $topo
    "--num_classes", "10"
    "--normalize", "True"
    "--decood_loss_code", $decood_loss_code
    "--lr", "$lr"
    "--feat_dim", "$feat_dim"
    "--num_rounds", "$n_rounds"
    "--dist", $dist
    "--local_epochs", "$local_epochs"
    "--num_clusters", "1"
    "--clustering", "label"
    "--noise_level", "0"
    "--data_type", "img"
    "--test_on_cosine", "True"
    "--sel_on_kappa", "False"
    "--dataset", $dataset
    "--num_trials", "$num_trials"
    "--num_clients", "$num_clients"
    "--feature_iid", "$feature_iid"
    "--label_iid", "$label_iid"
    "--local_bs", "$local_bs"
    "--tau", "$tau"
    "--weighted_adj_mat", "$weighted_adj_mat"
    "--backbone", $backbone
    "--use_imb_loss", "$use_imb_loss"
    "--alpha", "$alpha"
    "--data_dir", $data_dir
    "--save_folder_name", $save_folder_name
    "--params_dir", $params_dir
)

# Save both stdout and stderr to the log
& python @pythonArgs *>> $log_file

Write-Host "Log saved to: $log_file"