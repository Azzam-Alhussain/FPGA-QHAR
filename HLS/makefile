SHELL := /bin/zsh 

#######################################################
# medium is the zcu104
#~ PART=large #medium
PART=medium

main: hls bitstream

hls:
	source scripts/hls_lk.sh ${PART}
	source scripts/hls_cnn.sh ConvCNN_Quantized ${PART}
#~ 	mv HLS_lk LK_optical_flow/

bitstream:
	rm -rf vivado design_1.bit design_1.hwh
#~ 	vivado -mode batch -source scripts/two_stream_vivado_104.tcl -nojournal -nolog
	source scripts/bitsream_two_stream.sh  ${PART}
	rm -rf .Xil 
	cp vivado/vivado.runs/impl_1/design_1_wrapper.bit design_1_wrapper.bit
	cp vivado/vivado.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh design_1_wrapper.hwh

clean:
	rm -rf LK_Optical_Flow_Quantized/HLS_LK
	rm -rf ConvCNN_Quantized/HLS_PROJ
	rm -rf vivado*
