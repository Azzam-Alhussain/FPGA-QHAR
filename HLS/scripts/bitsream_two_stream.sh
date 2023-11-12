#!/bin/zsh

Red='\033[0;31m'
Cyan='\033[1;36m'
NC='\033[0m' # No Color

####### start script #######

SCALE=$1

case $SCALE in 
	small) SCRIPT="two_stream_vivado_z2.tcl" ;;
	medium) SCRIPT="two_stream_vivado_104.tcl" ;;
	large) SCRIPT="two_stream_vivado.tcl" ;;
	*) SCRIPT="xxx.tcl" ;;
esac 

echo "${Cyan}vivado_hls two_stream/$SCRIPT $NC"
vivado -mode batch -source scripts/$SCRIPT  -nojournal -nolog
