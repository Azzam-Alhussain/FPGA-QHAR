#include"top.h"



inline
data_t ***Init3DArray(int height, int width, int channel){
	data_t*** fm = new data_t**[height];
	for(int i = 0; i < height; i++){
		fm[i] = new data_t*[width];
		for(int j = 0; j < width; j++){
			fm[i][j] = new data_t[channel];
		}
	}

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			for(int k = 0; k < channel; k++){
				fm[i][j][k] = 0;
			}
		}
	}

	return fm;
}

inline
data_t ****Init4DArray(int kerSize, int outChannel, int inChannel){
	data_t**** arr = new data_t***[kerSize];
	for(int ky = 0; ky < kerSize; ky++){
		arr[ky] = new data_t**[kerSize];
		for(int kx = 0; kx < kerSize; kx++){
			arr[ky][kx] = new data_t*[outChannel];
			for(int o = 0; o < outChannel; o++){
				arr[ky][kx][o] = new data_t[inChannel];
			}
		}
	}

	for(int ky = 0; ky < kerSize; ky++){
		for(int kx = 0; kx < kerSize; kx++){
			for(int o = 0; o < outChannel; o++){
				for(int i = 0; i < inChannel; i++){
					arr[ky][kx][o][i] = 0;
				}
			}
		}
	}

	return arr;
}

/***********************************************
 *
 * Input feature map functions
 *
 ***********************************************/

template<unsigned inRow, unsigned inCol, unsigned inChannel>
void IFMInit(data_t ***ifm, char* mode){
	for(int i = 0; i < inRow; i++){
		for(int j = 0; j < inCol; j++){
			for(int k = 0; k < inChannel; k++){
				if(mode == "rand"){
#ifndef __SYNTHESIS__
					ifm[i][j][k] = rand()%256;
#endif
				}else if(mode == "order"){
					ifm[i][j][k] = i * inCol * inChannel + j * inChannel + k;
				}else{
					ifm[i][j][k] = 1;
				}
					//
			}
		}
	}

	if(mode == "file"){
		char buff[inRow*inCol*inChannel];
		data_t temp[inChannel][inRow][inCol];
		FILE *latfile;

#ifndef TEST_CASE
		sprintf(buff,"%s","ifm.dat");
#else
		sprintf(buff,"%s","ifm_1.dat");
#endif
		latfile=fopen(buff,"r");
		fread(&(temp[0][0][0]),sizeof(data_t),inRow*inCol*inChannel,latfile);
		fclose(latfile);

		for(int k = 0; k < inChannel; k++){
			for(int i = 0; i < inRow; i++){
				for(int j = 0; j < inCol; j++){
					ifm[i][j][k] = temp[k][i][j];
				}
			}
		}
	}
}

template<unsigned inRow, unsigned inCol, unsigned inChannel>
void IFMConvert(
		uint128 hw_input[inRow*inCol*inChannel],
		data_t ***ifm,
		int inTiles){
	int ifmOffset = 0;
	// write to a linear memory array
	for(int inTid = 0; inTid < inTiles; inTid++){
		for(int i = 0; i < inRow; i++){
			for(int j = 0; j < inCol; j++){
				ifmOffset = (inTid*inRow*inCol + i*inCol + j) * WORDS_PER_LINE;
				for(int wrd = 0; wrd < WORDS_PER_LINE; wrd++){
					for(int k = 0; k < WORD_COUNT; k++){
//						printf("(k+1)*PREC = %d, k*PREC = %d \n", (k+1)*PREC, k*PREC);
						hw_input[ifmOffset + wrd].range((k+1)*PREC-1, k*PREC)
								= ifm[i][j][inTid*Ti + wrd*WORD_COUNT + k];
					}
				}
			}
		}
	}
}

template<unsigned height, unsigned width, unsigned inChannel>
void IFMMonitor(data_t ***act, int type){

	if(type == 1){

		for(int k = 0; k < inChannel; k++){
			cout << "Channel index = " << k << endl;
			if(k > 35) break;
			for(int i = 0; i < height; i++){
				for(int j = 0; j < width; j++){
					cout << (int)act[i][j][k] << ", ";
				}
				cout << endl;
			}
			cout << endl;
		}
	}
}

template<unsigned inRow, unsigned inCol, unsigned inChannel>
void IFMMonitorLinear(
		uint128 hw_input[inRow*inCol*inChannel], int row, int col, int inTiles, int type){

	if(type == 1){
		int offset = 0;
			for(int t = 0; t < inTiles; t++){
				for(int i = 0; i < row; i++){
					cout << "Tile: " << t << ", row: " << i << endl;
					for(int j = 0; j < col; j++){
						offset = (t*row*col + i*col + j)*WORDS_PER_LINE;
						for(int wrd = 0; wrd < WORDS_PER_LINE; wrd++){
							for(int k = 0; k < WORD_COUNT; k++){
								cout << (int)hw_input[offset + wrd].range((k+1)*PREC-1, k*PREC) << ", ";
							}
							cout << " ||| ";
						}
						cout << endl;
					}
					cout << endl;
				}
			}
	}

}

inline
void IFMMonitorTile(
		uintTi act[MAX_TILE_IN_HEIGHT][MAX_TILE_IN_WIDTH],
	int tidY, int tidX, int tidIn,
	int inTr, int inTc){

	printf("tidY = %d, tidX = %d, tidIn = %d \n", tidY, tidX, tidIn);

	for(int i = 0; i < Ti; i++){
		cout << "Channel id = " << i << endl;
		for(int y = 0; y < inTr; y++){
			for(int x = 0; x < inTc; x++){
				cout << (int)act[y][x].range((i+1)*PREC-1, i*PREC) << ", ";
			}
			cout << endl;
		}
	}

	cout << endl;
}

/***********************************************
 *
 * Weight functions
 *
 ***********************************************/

template<unsigned kerSize, unsigned outChannel, unsigned inChannel>
void WGTInit(data_t ****wgt, char* mode, char* dataMode){

	if(mode == "channel"){

		for(int k_out = 0; k_out < outChannel; k_out++){
			for(int i = 0; i < kerSize; i++){
				for(int j = 0 ; j < kerSize; j++){
					for(int k_in = 0; k_in < inChannel; k_in++){
						if(dataMode == "rand"){
#ifndef __SYNTHESIS__
							wgt[i][j][k_out][k_in] =rand()%256;
#endif
						}else if(dataMode == "order"){
							wgt[i][j][k_out][k_in] =
							k_out * kerSize * kerSize * inChannel + i * kerSize * inChannel + j * inChannel + k_in;
						}else{
							wgt[i][j][k_out][k_in] = 1;
						}
					}
				}
			}
		}
	}else if(mode == "row"){
		for(int k_out = 0; k_out < outChannel; k_out++){
			for(int k_in = 0; k_in < inChannel; k_in++){
				for(int i = 0; i < kerSize; i++){
					for(int j = 0; j < kerSize; j++){
						wgt[i][j][k_out][k_in] =
								k_out * inChannel * kerSize * kerSize + k_in * kerSize * kerSize + i * kerSize + j;
					}
				}
			}
		}
	}

	if(dataMode == "file"){
		char buff[kerSize*kerSize*outChannel*inChannel];
		data_t temp[outChannel][inChannel][kerSize][kerSize];
		FILE *latfile;

#ifndef TEST_CASE
		sprintf(buff,"%s","wgt.dat");
#else
		sprintf(buff,"%s","wgt_1.dat");
#endif
		latfile=fopen(buff,"r");
		fread(&(temp[0][0][0][0]),sizeof(data_t),kerSize*kerSize*outChannel*inChannel,latfile);
		fclose(latfile);

		for(int o = 0; o < outChannel; o++){
			for(int i = 0; i < inChannel; i++){
				for(int ky = 0; ky < kerSize; ky++){
					for(int kx = 0; kx < kerSize; kx++){
//						printf("o = %d, i = %d, ky = %d, kx = %d", o,i,ky,kx);
						wgt[ky][kx][o][i] = temp[o][i][ky][kx];
					}
				}
			}
		}
	}
}

template<unsigned kerSize, unsigned outChannel, unsigned inChannel>
void WGTConvert(
		uint128 hw_wgt[kerSize*kerSize*outChannel*inChannel],
	data_t**** weight,
	int outTiles, int inTiles
	){

	// TODO: currently kernel spatial major, compare with output channel major
	for(int tidOut = 0; tidOut < outTiles; tidOut++){
		for(int tidIn = 0; tidIn < inTiles; tidIn++){
			for(int o = 0; o < To; o++){
				for(int ky = 0; ky < kerSize; ky++){
					for(int kx = 0; kx < kerSize; kx++){
						for(int wrd =0; wrd < WORDS_PER_LINE; wrd++){
							for(int i = 0; i < WORD_COUNT; i++){
								hw_wgt[tidOut*To*inTiles*kerSize*kerSize*WORDS_PER_LINE // tidOut offset
									+tidIn*To*kerSize*kerSize*WORDS_PER_LINE // tidIn offset
									+o*kerSize*kerSize*WORDS_PER_LINE // filter offset
									+ky*kerSize*WORDS_PER_LINE // filter row offset
									+kx*WORDS_PER_LINE  // filter column in a row offset
									+wrd].range((i+1)*PREC-1, i*PREC) =
								weight[ky][kx][tidOut*To + o][tidIn*Ti + wrd*WORD_COUNT + i];
							}
						}
					}
				}
			}
		}
	}
}

template<unsigned kerSize, unsigned outChannel, unsigned inChannel>
void WGTMonitor(data_t**** weight, int type){

	if(type == 1){
		for(int o = 0; o < outChannel; o++){
			cout << "Output channel index = " << o << endl;
//			if(o < 32) continue;
			for(int ky = 0; ky < kerSize; ky++){
				for(int kx = 0; kx < kerSize; kx++){
					for(int i = 0; i < inChannel; i++){
						cout << (int)weight[ky][kx][o][i] << ", ";
					}
					cout << endl;
				}
			}
			cout << endl;
		}
	}else if(type == 2){
		for(int o = 0; o < outChannel; o++){
			cout << "Output channel index = " << o << endl;
		//	if(o < 32) continue;
			for(int i = 0; i < inChannel; i++){
				for(int ky = 0; ky < kerSize; ky++){
					for(int kx = 0; kx < kerSize; kx++){
						cout << (int)weight[ky][kx][o][i] << ", ";
					}
					cout << endl;
				}
				cout << endl;
			}
		}
	}
}

template<unsigned kerSize, unsigned outChannel, unsigned inChannel>
void WGTMonitorLinear(
		uint128 hw_wgt[kerSize*kerSize*outChannel*inChannel],
	int outTiles, int inTiles, int type){

	if(type == 1){
		for(int tidOut = 0; tidOut < outTiles; tidOut++){
			if(tidOut < 1) continue;
			for(int tidIn = 0; tidIn < inTiles; tidIn++){
				for(int o = 0; o < To; o++){
					printf("\n tidOut = %d, tidIn = %d, FilterID = %d\n", tidOut, tidIn, o+tidOut*To);
					for(int ky = 0; ky < kerSize; ky++){
						for(int kx = 0; kx < kerSize; kx++){
							for(int wrd = 0; wrd < WORDS_PER_LINE; wrd++){
								for(int i = 0; i < WORD_COUNT; i++){
									cout << (int)hw_wgt[
										tidOut*To*inTiles*kerSize*kerSize*WORDS_PER_LINE // tidOut offset
										+tidIn*To*kerSize*kerSize*WORDS_PER_LINE // tidIn offset
										+o*kerSize*kerSize*WORDS_PER_LINE // filter offset
										+ky*kerSize*WORDS_PER_LINE // filter row offset
										+kx*WORDS_PER_LINE  // filter column in a row offset
										+wrd
									].range((i+1)*PREC-1, i*PREC) << ", ";
								}
								cout << " ||| ";
							}
							cout << endl;
						}
					}
				}
			}
		}
	}
}

inline
void WGTMonitorTile(
		uintTi wgt[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][To],
	int tidOut, int tidIn, int kerSize){

	for(int o = 0; o < To; o++){
		printf("tidOut = %d, tidIn = %d, FilterID = %d \n", tidOut, tidIn, o);
		for(int ky = 0; ky < kerSize; ky++){
			for(int kx = 0; kx < kerSize; kx++){
				for(int i = 0; i < Ti; i++){
					cout << (int)wgt[ky][kx][o].range((i+1)*PREC-1, i*PREC) << ", ";
				}
				cout << endl;
			}
		}
		cout << endl;
	}
}

/***********************************************
 *
 * Output feature map functions
 *
 ***********************************************/
template<unsigned outRow, unsigned outCol, unsigned outChannel>
void ReadOFMFromFile(data_t ***ofm){
	char buff[outRow*outCol*outChannel];
	data_t temp[outChannel][outRow][outCol];
	FILE *latfile;

#ifndef TEST_CASE
		sprintf(buff,"%s","ofm.dat");
#else
		sprintf(buff,"%s","ofm_1.dat");
#endif
	latfile=fopen(buff,"r");
	fread(&(temp[0][0][0]),sizeof(data_t),outRow*outCol*outChannel,latfile);
	fclose(latfile);

	for(int k = 0; k < outChannel; k++){
		for(int i = 0; i < outRow; i++){
			for(int j = 0; j < outCol; j++){
				ofm[i][j][k] = temp[k][i][j];
			}
		}
	}

}


template<unsigned outChannel>
void OFMConvert(
		data_t ***hw_result,
		uint128 *hw_output,
		int outRow, int outCol){

	int offset = 0;
	int tileNumOut = divide_ceil(outChannel, To);
	int wordPerOutLine = To/WORD_COUNT;
	for(int tidOut = 0; tidOut < tileNumOut; tidOut++){
		for(int y = 0; y < outRow; y++){
			for(int x = 0; x < outCol; x++){
				offset = (tidOut*outRow*outCol + y*outCol + x)*wordPerOutLine;
				for(int wrd = 0; wrd < wordPerOutLine; wrd++){
					for(int i = 0; i < WORD_COUNT; i++){
						hw_result[y][x][tidOut*To + wrd*WORD_COUNT + i] =
						hw_output[offset + wrd].range((i+1)*PREC-1, i*PREC);
//						printf("hw_result[%d][%d][%d] = %d \n",
//								y,x,tidOut*To + wrd*WORD_COUNT + i,
//								(int)hw_output[offset + wrd].range((i+1)*PREC-1, i*PREC));

					}
				}
			}
		}
	}
}

template<unsigned outChannel>
void OFMMonitor(data_t ***hw_result,int outRow, int outCol){

	for(int o = 0; o < outChannel; o++){
		cout << endl;
		cout << "Output Channel Index = " << o << endl;
		for(int x = 0; x < outRow; x++){
			for(int y = 0; y < outCol; y++){
				cout << (int)hw_result[y][x][o] << ", ";
			}
			cout << endl;
		}
	}
}

inline
void OFMMonitorLinear(
		uint128 *hw_input, int outRow, int outCol, int outChannel){

	for(int i = 0; i < outRow*outCol*outChannel/WORD_COUNT; i++){

		for(int w = 0; w < WORD_COUNT; w++){
			cout << (int)hw_input[i].range((w+1)*PREC-1, w*PREC) << ", ";
		}
		cout << endl;
	}
}


template<typename T>
void OFMMonitorTile(
		psum_t psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH][To], int Tr, int Tc
		, int tidX, int tidY, int tidOut
){
	for(int o = 0; o < To; o++){
		printf("TidX = %d, TidY = %d, TidOut = %d, Channel index = %d \n", tidX, tidY, tidOut, o);
		for(int i = 0; i < Tr; i++){
			for(int j = 0; j < Tc; j++){
				cout << (T)psum_output[i][j][o] << ", ";
			}
			cout << endl;
		}
		cout << endl;
	}
}




template<unsigned inRow, unsigned inCol, unsigned kerSize, unsigned outChannel, unsigned inChannel, unsigned stride>
void SWConv(
		data_t ***sw_result,
		data_t ifm[inRow][inCol][inChannel],
		data_t weight[kerSize][kerSize][outChannel][inChannel]
		){

	int outRow = divide_ceil(inRow, stride);
	int outCol = divide_ceil(inCol, stride);
	cout << "kerSize = " << -1 <<endl;
 	for(int i = 0; i < outRow; i++){
 		for(int j = 0; j < outCol; j++){
 			for(int k_out = 0; k_out < outChannel; k_out++){
 				psum_t partial_sum = 0;
 				for(int k_in = 0; k_in < inChannel; k_in++){
 					for(int k_i = -1; k_i < kerSize-1; k_i++){
 						for(int k_j = -1; k_j < kerSize-1; k_j++){
 							if((i*stride+k_i >= 0 && j*stride+k_j >= 0) && (i*stride+k_i < inRow && j*stride+k_j < inCol)){
 								partial_sum += weight[k_i+1][k_j+1][k_out][k_in]
 											* ifm[i*stride+k_i][j*stride+k_j][k_in];
 							}
 						}
 					}
 				}
 				sw_result[i][j][k_out] = partial_sum;
 //
 			}
 		}
 	}
}

inline
void SW_Pooling(data_t ***sw_result, data_t ***sw_conv_result, int row, int column, int channel, int poolWin){

	for(int z = 0; z < channel; z++){
		for(int y = 0; y < row; y+=poolWin){
			for(int x = 0; x < column; x+=poolWin){

				data_t temp = sw_conv_result[y][x][z];
				for(int i = 0; i < poolWin; i++){
					for(int j = 0; j < poolWin; j++){
						if(sw_conv_result[y+i][x+j][z] > temp){
							temp = sw_conv_result[y+i][x+j][z];
						}
					}
				}
				sw_result[y/poolWin][x/poolWin][z] = temp;
			}
		}
	}
}

template<unsigned inRow, unsigned inCol, unsigned inChannel, unsigned outChannel, unsigned kerSize>
void SW_FCN(
		data_t ***sw_result,
		data_t ***act,
		data_t weight[kerSize][kerSize][outChannel][inChannel]){

	for(int o = 0; o < outChannel; o++){
		psum_t acc = 0;
		for(int i = 0; i < inChannel; i++){
			acc += weight[0][0][o][i] * act[0][0][i];

		}
		sw_result[0][0][o] = acc;
	}
}

int main(){

	srand(30);

#ifndef TEST_CASE
	float multiplier = 0.002746367361396551;
	data_t zpW = 128, zpX = 7, zpXNext = 7;
	const int inRow = 32, inCol = 32, outRow = 32, outCol = 32;
	const int inChannel=3, outChannel=32;
	const int poolWin = 1;

#else
	float multiplier = 0.0019095869502052665;
	data_t zpW = 109, zpX = 5, zpXNext = 6;
	const int inRow = 16, inCol = 16, outRow = 8, outCol = 8;
	const int inChannel=32, outChannel=64;
	const int poolWin = 2;
#endif

	const int Tr = 8, Tc = 8;
	const int kerSize = 3;
	const int stride = 1;


	int inTiles = divide_ceil(inChannel, Ti);
	int outTiles = divide_ceil(outChannel, To);

	bool isFCN = (inRow == 1 && inCol == 1)? true:false;

	char* dataMode = (char*)"file";
	data_t ***act = Init3DArray(inRow, inCol, (inChannel>WORD_COUNT)?inChannel:WORD_COUNT);
	data_t ****weight = Init4DArray(kerSize, outChannel, inChannel);
	data_t ***sw_result = Init3DArray(outRow, outCol, outChannel);
	data_t ***hw_result = Init3DArray(outRow, outCol, outChannel);

	// convert to hardware data format
	uint128 *hw_input = new uint128[inRow*inCol*inChannel];
	uint128 *hw_wgt = new uint128[kerSize*kerSize*outChannel*inChannel];
	uint128 *hw_output = new uint128[outRow*outCol*outChannel];

	// initialize activation
	IFMInit<inRow, inCol, inChannel>(act, dataMode);
	IFMConvert<inRow, inCol, inChannel>(hw_input, act, inTiles);
	IFMMonitor<inRow, inCol, inChannel>(act, 0);
	IFMMonitorLinear<inRow, inCol, inChannel>(hw_input, inRow, inCol, inTiles, 0);

//	// initialize weight
	WGTInit<kerSize, outChannel, inChannel>(weight, (char*)"channel", dataMode);
	WGTConvert<kerSize, outChannel, inChannel>(hw_wgt, weight, outTiles, inTiles);
	WGTMonitor<kerSize, outChannel, inChannel>(weight, 0);
	WGTMonitorLinear<kerSize, outChannel, inChannel>(hw_wgt,outTiles, inTiles, 0);

	//read software output feature map
	ReadOFMFromFile<outRow, outCol, outChannel>(sw_result);

	// hardware conv
	DoCompute(hw_input, hw_output, hw_wgt,
	 		inRow, inCol, inChannel, outChannel,
	 		Tr, Tc, kerSize, stride, poolWin,
			multiplier, zpX, zpW, zpXNext);
//	OFMMonitorLinear(hw_output, outRow, outCol, outChannel);

	OFMConvert<outChannel>(hw_result, hw_output, outRow, outCol);

//	OFMMonitor<outChannel>(hw_result, outRow, outCol);

	int err = 0;
 	for(int k = 0; k < outChannel; k++){
 		printf("================== channel = %d ===============\n", (k));
 		for(int i = 0; i < outRow; i++){
 			for(int j = 0; j < outCol; j++){
 				if(sw_result[i][j][k] != hw_result[i][j][k]){
 					err++;
 				}
 				cout << sw_result[i][j][k] << ":" << hw_result[i][j][k] << ", ";
// 				cout << sw_result[i][j][k] << ", ";
 			}
 			printf("\n");
 		}
 	}
 	printf("==================== errors = %d ===========================\n", err);

	return err;

}
