#!/bin/bash

ssh -f -N  mlflow_user@35.223.113.101 -L 0.0.0.0:5050:35.223.113.101:8000 -o TCPKeepAlive=yes
ssh -f -N  mlflow_user@35.223.113.101 -L 0.0.0.0:5054:35.223.113.101:22 -o TCPKeepAlive=yes
