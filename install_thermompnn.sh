#!/bin/bash
################## ThermoMPNN installation script

# Default value for pkg_manager
pkg_manager='conda'
cuda='11.7'

# Define the short and long options
OPTIONS=p:c:
LONGOPTIONS=pkg_manager:,cuda:

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

# Example usage of the parsed variables
echo -e "Package manager: $pkg_manager"
echo -e "CUDA: $cuda"

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths needed for installation and check for conda installation
install_dir=$(pwd)
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

### Create base environment
echo -e "Installing thermoMPNN environment\n"
$pkg_manager create --name thermoMPNN-ph python=3.10 -y || { echo -e "Error: Failed to create thermoMPNN conda environment"; exit 1; }
conda env list | grep -w 'thermoMPNN-ph' >/dev/null 2>&1 || { echo -e "Error: Conda environment 'thermoMPNN-ph' does not exist after creation."; exit 1; }

# Load newly created environment
echo -e "Loading thermoMPNN environment\n"
source ${CONDA_BASE}/bin/activate thermoMPNN-ph || { echo -e "Error: Failed to activate the thermoMPNN environment."; exit 1; }
[ "$CONDA_DEFAULT_ENV" = "thermoMPNN-ph" ] || { echo -e "Error: The thermoMPNN environment is not active."; exit 1; }
echo -e "thermoMPNN environment activated"

# Set memory-efficient solver
export CONDA_SOLVER=libmamba

# Install packages in smaller groups
echo "Installing packages..."
if [ -n "$cuda" ]; then
    CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install -y \
        pytorch pytorch-cuda="$cuda" torchvision torchaudio \
        torchmetrics pytorch-lightning biopython wandb mmseqs2 \
        tqdm pandas numpy omegaconf joblib -c pytorch -c nvidia -c conda-forge -c bioconda || { echo -e "Error: Failed to install packages."; exit 1; }
else
    $pkg_manager install -y \
        pytorch torchvision torchaudio \
        torchmetrics pytorch-lightning biopython wandb mmseqs2 \
        tqdm pandas numpy omegaconf joblib -c pytorch -c conda-forge -c bioconda || { echo -e "Error: Failed to install packages."; exit 1; }
fi

# make sure all required packages were installed
required_packages=(pytorch torchvision torchaudio torchmetrics pytorch-lightning biopython wandb mmseqs2 tqdm pandas numpy omegaconf joblib)
missing_packages=()

# Check each package
for pkg in "${required_packages[@]}"; do
    conda list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
done

# If any packages are missing, output error and exit
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

# finish
conda deactivate
echo -e "thermoMPNN environment set up\n"

############################################################################################################
############################################################################################################
################## cleanup
echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y
echo -e "$pkg_manager cleaned up\n"

################## finish script
t=$SECONDS 
echo -e "Successfully finished thermoMPNN installation!\n"
echo -e "Activate environment using command: \"$pkg_manager activate thermoMPNN-ph\""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."