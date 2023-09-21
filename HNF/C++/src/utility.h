//
// Created by batman on 26.08.23.
//

#include <iostream>
#include <NTL/vector.h>
#include <NTL/ZZ.h>
#include <NTL/matrix.h>
#include <NTL/ZZ_p.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/mat_ZZ.h>
#include <NTL/tools.h>
#include <NTL/HNF.h>
#include <cmath>
#include <typeinfo>
#include <NTL/RR.h>
#include <NTL/mat_RR.h>

using namespace std;
using namespace NTL;

#define RANDOM_SIZE_BITS 8

void test(int);
void getRandomMatrix(Mat<ZZ>& dst, unsigned long m, unsigned long n);
void getRandomVector(Vec<ZZ>& dst, unsigned long n);
void CRTOnArray(ZZ& dst, Vec<ZZ>& elements, Vec<ZZ>& moduli, unsigned long n);
void HNFModD(Mat<ZZ>&, Mat<ZZ>&, ZZ& det);
void hnfModD(Mat<ZZ>&, Mat<ZZ>&, ZZ& det);
void addRowToMatrix(Mat<ZZ>&, Vec<ZZ>&);
void addColumnToMatrix(Mat<ZZ>&, Mat<ZZ>&, Vec<ZZ>&);
void getPrincipalMinor(Mat<ZZ>&, Mat<ZZ>&, unsigned long);
void get_a_t(Vec<ZZ>&, Mat<ZZ>&, long);
void get_a_column(Vec<ZZ>&, Mat<ZZ>&, long);
void get_e_i(Vec<ZZ>& dst, long n, long idx);
ZZ getDiagProd(Mat<ZZ>& m);
ZZ getMaxColumnLength(Mat<ZZ>& m);
void addRow(Vec<ZZ>& dst, Mat<ZZ>& B, Mat<ZZ>& H_B, Vec<ZZ>& a_t);
void addRowWithoutCRT(Vec<ZZ>& dst, Mat<ZZ>& B, Mat<ZZ>& H_B, Vec<ZZ>& a_t);
void addColumn(Mat<ZZ>& dst, Mat<ZZ>& A, Vec<ZZ>& b);
void hnf(Mat<ZZ>& dst, Mat<ZZ>& B);
void decomposeForHeuristic(Mat<ZZ>& B,Vec<ZZ>& b_t,Vec<ZZ>& c,Vec<ZZ>& d, ZZ& a_1, ZZ& a_2, Mat<ZZ>& M);
void hnf_heuristic(Mat<ZZ>& H, Mat<ZZ>& A);
