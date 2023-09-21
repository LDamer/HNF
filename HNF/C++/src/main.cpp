#include <iostream>
#include "utility.h"

using std::cout;
using std::endl;
using namespace NTL;

int main(void) {
    Mat<ZZ> random, H;
    random.SetDims(3,3);
    //getRandomMatrix(random, 100, 100);
    random[0][0] = ZZ(220);random[0][1] = ZZ(210);random[0][2] = ZZ(40);
    random[1][0] = ZZ(0);random[1][1] = ZZ(140);random[1][2] = ZZ(203);
    random[2][0] = ZZ(215);random[2][1] = ZZ(128);random[2][2] = ZZ(130);
    ZZ d = abs(determinant(random));
    cout << "random\n" << random << endl;
    cout << "start" << endl;
    double t1 = GetTime();
    hnf(H, random);
    double t2 = GetTime();
    cout << "H=\n" << H << endl;
    cout << t2 - t1 << " seconds" << endl;
    //////////////////////
    /*Mat<ZZ> A, H;
    A.SetDims(2,2);
    A(1,1) = ZZ(50);
    A(1,2) = ZZ(2889);
    A(2,1) = ZZ(130);
    A(2,2) = ZZ(75115);
    cout << A << endl;
    ZZ g = abs(determinant(A));
    HNF(H, A, g);
    cout << H << endl;*/
    //////////////////////////////////////
    return 0;
}
