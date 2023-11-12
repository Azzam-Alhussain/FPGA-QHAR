
//Use this to define the maximum width of ap_int variables
#define AP_INT_MAX_W	1024*2	//32768

#include<stdio.h>
#include<iostream>
#include<ap_int.h>
#include<ap_fixed.h>
#include<hls_stream.h>
#include<hls_math.h>
#include<stdint.h>
#include<math.h>

using namespace std;
using namespace hls;


#define ZCU104
#define TEST_CASE

#define PREC 8
#define ACC_PREC 32

#ifdef CSIM_FLAG
#define MAX_TILE_OUT_HEIGHT 3
#define MAX_TILE_OUT_WIDTH 3

#else
#define MAX_TILE_OUT_HEIGHT 96
#define MAX_TILE_OUT_WIDTH 96
#endif

#define MAX_STRIDE 2
#define MAX_KERNEL_SIZE 7
#define MAX_TILE_IN_HEIGHT MAX_TILE_OUT_HEIGHT*MAX_STRIDE + (MAX_KERNEL_SIZE-MAX_STRIDE)
#define MAX_TILE_IN_WIDTH MAX_TILE_OUT_WIDTH*MAX_STRIDE + (MAX_KERNEL_SIZE-MAX_STRIDE)


#ifdef ZCU104

#define To 16 // output channel
#define Ti 16 // input channel

#define ToC 64/To // output channel
#define TiC 64/Ti // input channel


#else
#define To 32 // output channel
#define Ti 32 // input channel
#endif


#define DATAWIDTH PREC*Ti
#define WORD_COUNT (int)(DATAWIDTH/PREC)


#define WORDS_PER_LINE Ti/WORD_COUNT

#define WGT_CHUNK_NUM 1
#define HARDCODE_VAL 1

typedef ap_ufixed<8,8,AP_RND_CONV,AP_SAT> clamp_round_t;
typedef ap_uint<PREC> data_t;
typedef ap_uint<ACC_PREC> psum_t;
typedef ap_uint<DATAWIDTH> uint128;
typedef ap_uint<Ti*PREC> uintTi;
typedef ap_uint<To*PREC> uintTo;
typedef ap_uint<To*ACC_PREC> uintAcc;

/*
 * block data
 */
template<typename T, unsigned WIDTH>
struct row_t{
	T data[WIDTH];
};

template<typename T, unsigned HEIGHT, unsigned WIDTH>
struct block_t{
	T data[HEIGHT][WIDTH];
};

void DoCompute(
		uint128 *ifm,
		uint128 *ofm,
		uint128 *raw_wgt,
		int inRow, int inCol, int inChannel, int outChannel,
		int Tr, int Tc, int kerSize, int stride, int poolWin,
		float multiplier, data_t zpX, data_t zpW, data_t zpXNext);

void Convolution(uint128 *ifm,
		uint128 *ofm,
		uint128 *raw_wgt,
		int inRow, int inCol, int inChannel, int outChannel,
		int Tr, int Tc, int kerSize, int stride,
		int tileNumX, int tileNumY, int tileNumIn, int tileNumOut,
		uintTi act[MAX_TILE_IN_HEIGHT][MAX_TILE_IN_WIDTH],
		uintAcc psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH]);

void LoadActivation(
		uint128 *ifm,
		uintTi act[MAX_TILE_IN_HEIGHT][MAX_TILE_IN_WIDTH],
		int inRow, int inCol,
		int Tr, int Tc, int anchorY, int anchorX, int offset,
		int inTr, int inTc);

void LoadWeight(
		uint128 *raw_wgt,
		uintTi wgt[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][To],
		int tidIn, int tidOut, int tileNumIn,
		int kerSize);

void WriteOutput(
		uint128 *ofm,
		psum_t psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH][To],
		float multiplier, data_t zpXNext,
		int tidY, int tidX, int tidOut,
		int Tr, int Tc, int inRow, int inCol,
		int poolWin);

void WriteDRAM(
		uint128 *output, psum_t buffer[To],
		int wordOffset, float multiplier, data_t zpXNext
		);

void PoolingOutput(
		uint128 *output,
		uintAcc psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH],
		int Tr, int Tc, int outTr, int outTc);

inline
int divide_ceil(int divident, int divisor){
#pragma HLS INLINE
	if(divident < divisor){
		return 1;
	}
	return (divident%divisor == 0)?divident/divisor: (int)(divident/divisor)+1;
}
inline
int minimum(int a, int b){
	if(a >= b){
		return b;
	}
	return a;
}

inline
bool notBoundary(int i, int j, int anchorY, int anchorX , int inRow, int inCol){
	int ptrX = anchorX + j;
	int ptrY = anchorY + i;
	return ((ptrX >= 0) && (ptrY >= 0) && (ptrX < inCol) && (ptrY < inRow));
}



/***********************************************
 *
 * Convolution
 *
 ***********************************************/


template<typename ACC_T>
psum_t PE(ACC_T psum,
		uintTi act,
		uintTi wgt,
		data_t zpX,
		data_t zpW,
		int enableBit){
#pragma HLS INLINE OFF

	for(int i = 0; i < Ti; i++){
#pragma HLS UNROLL
		if(i < enableBit){
			psum += ((ap_int<9>)(act.range((i+1)*PREC-1, i*PREC)) - zpX) *
					((ap_int<9>)(wgt.range((i+1)*PREC-1, i*PREC)) - zpW);
		}
	}
	return psum;
}

template<unsigned numPE>
void HWConv(
		uintTi wgt[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][To],
		uintTi act[MAX_TILE_IN_HEIGHT][MAX_TILE_IN_WIDTH],
		psum_t psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH][To],
		data_t zpX, data_t zpW,
		int k1, int k2, int Tr, int Tc, int stride, int padding,
		int anchorY, int anchorX,
		int inRow, int inCol, int enableBit
){
#pragma HLS ALLOCATION instances=PE limit=numPE function

#pragma HLS ARRAY_PARTITION variable=wgt dim=3 complete


	CONV_LOOP_KI:for(int ki = -padding; ki < k1-padding; ki++){
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
		CONV_LOOP_KJ:for(int kj = -padding; kj < k2-padding; kj++){
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
			CONV_LOOP_TR:for(int r = 0; r < Tr; r++){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
				CONV_LOOP_TC:for(int c = 0; c < Tc; c++){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
#pragma HLS PIPELINE
					if(notBoundary(r+ki+padding, c+kj+padding, anchorY, anchorX, inRow, inCol)){

						CONV_LOOP_TO_COMP:for(int o = 0; o < To; o++){
//~ #pragma HLS UNROLL
							psum_output[r][c][o] = PE<psum_t>(
									psum_output[r][c][o],
								act[r+ki+padding][c+kj+padding],
								wgt[ki+padding][kj+padding][o],
								zpX, zpW, enableBit
							);
						}
					}
				}
			}
		}
	}
}

template<typename T>
void InitPSUM(T psum_output[MAX_TILE_OUT_HEIGHT][MAX_TILE_OUT_WIDTH][To]){
	// reset psum
	for(int i = 0; i < MAX_TILE_OUT_HEIGHT; i++){
		for(int j = 0; j < MAX_TILE_OUT_WIDTH; j++){
#pragma HLS PIPELINE
			for(int o = 0; o < To; o++){
				#pragma HLS UNROLL
				psum_output[i][j][o] = 0;
			}
		}
	}
}


