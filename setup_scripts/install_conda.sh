
set -e
echo "Did you run bash Miniconda3-latest-Linux-x86_64.sh first?"
echo "If you did, did you also create a new conda env by running conda create --name myenv scipy=1.0.0? [yes/no]"

read response

if [ "$response" == "yes" ]; then
  echo "Installing.. "
  conda install --file ../conda.txt

else
  echo "exiting"
  exit 1
fi