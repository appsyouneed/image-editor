1. Connect to vps
2. Clone this repo
3. cd to /image-editor
4. Run bash setup.sh

To run later:

ssh-keygen -f '/root/.ssh/known_hosts' -R 'ip_address'

ssh root@ip_address

cd image-editor

python3 app.py --share
