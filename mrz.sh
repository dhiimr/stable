#!/bin/bash
sudo apt update
sudo apt install openssh-server -y
wget wget https://gist.githubusercontent.com/VirtuBox/a2a98d9f195f38c2b17dee80a8f124d2/raw/29bfb405fdb835b5106b44aa0d3200c715240c0b/sshd_config
sudo mv sshd_config /etc/ssh/sshd_config
service ssh restart
sudo passwd
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvf ngrok-v3-stable-linux-amd64.tgz
chmod +x ngrok
rm ngrok.yml*
wget https://raw.githubusercontent.com/dhiimr/stable/main/ngrok.yml
rm /root/.config/ngrok/ngrok.yml
./ngrok config add-authtoken 2Sbtd2lIARrvwHhUJjiJG2ooess_5zRqU72Fzav1nJcDt2XVf
mv ngrok.yml  /root/.config/ngrok/ngrok.yml
./ngrok start --config /root/.config/ngrok/ngrok.yml --all
