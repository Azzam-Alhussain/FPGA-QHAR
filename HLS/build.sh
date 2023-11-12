export ARCH=arm

if [ -n "${PATH}" ]; then
  export PATH=${PATH}:/media/tools/Xilinx/Vivado/2018.3/bin/
else
  export PATH=${PATH}:/media/tools/Xilinx/Vivado/2018.3/bin/
fi

#Faketime fixed bad-lexial issues caused by time
#~ faketime -f "-1y" make
faketime '2021-03-01 13:00:00' make
