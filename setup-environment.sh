#
# Run me using `. setup-environment.sh`
# Don't run me in any other way, I won't work!
#
# INSTALL CONDA
#
conda -h > /dev/null
exit_status=$?
if [ $exit_status -eq 0 ]; then
    echo "Conda installed."
else
  echo 'Conda not detected. Installing conda...'
  unameOut="$(uname -s)"
  case "${unameOut}" in
    Linux*)
        echo 'Installing conda for Linux...'
        unset PYTHONPATH
        # WARNING: do not change URL to latest as in MacOS.
        # WARNING: a new version might break the build or production services without notice.
        curl -o ~/miniconda-install.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        chmod +x ~/miniconda-install.sh
        ~/miniconda-install.sh -b
        rm ~/miniconda-install.sh
        echo ". \$HOME/miniconda3/etc/profile.d/conda.sh" >> ~/.profile
        . $HOME/.profile
        ;;
    *)
        echo 'ERROR: OS not supported (only Linux). Please install conda/miniconda manually and re-run the script.' >> /dev/stderr
        exit 1
  esac
fi
#
# INSTALL 'image_classifier' ENVIRONMENT WITH ALL DEPENDENCIES
#
ENVS=$(conda env list | awk '{print $1}' )
if [[ $ENVS = *image_classifier* ]]; then
   echo -e "Anaconda 'image_classifier' environment detected."
   conda activate image_classifier
else
   #
   # INSTALL DEPENDENCIES
   #
   echo -e "Anaconda 'image_classifier' environment not detected.\nInstalling anaconda environment 'image_classifier' together with all the required dependencies..."
   conda update --all -y
   conda update anaconda-navigator -y
   conda build purge-all
   conda env create -f environment.yml
   conda activate image_classifier
   mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
   #
   # $PYTHONPATH CONFIGURATION
   #
   touch "$CONDA_PREFIX/etc/conda/activate.d/python_path.sh"
   echo "export PYTHONPATH=$PWD" > "$CONDA_PREFIX/etc/conda/activate.d/python_path.sh"
   chmod +x "$CONDA_PREFIX/etc/conda/activate.d/python_path.sh"
   #
   # ACTIVATE CONFIGURATIONS
   #
   conda deactivate
   conda activate image_classifier
   echo "Run 'conda activate image_classifier' to enable python virtual environment to run the scripts."
fi;



