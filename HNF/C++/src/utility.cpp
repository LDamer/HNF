//
// Created by batman on 26.08.23.
//

#include "utility.h"

void test(int a){
    cout << a << endl;
}

void getRandomMatrix(Mat<ZZ>& dst, unsigned long m, unsigned long n){
    dst.SetDims(m, n);
    for(unsigned long i = 0; i < m; i++){
        for(unsigned long j = 0; j < n; j++){
            dst[i][j] = RandomBits_ZZ(RANDOM_SIZE_BITS);
        }
    }
}

void getRandomVector(Vec<ZZ>& dst, unsigned long n){
    dst.SetLength(n);
    for(int i = 0; i < n; i++){
        dst[i] = RandomBits_ZZ(RANDOM_SIZE_BITS);
    }
}


void CRTOnArray(ZZ& dst, Vec<ZZ>& elements, Vec<ZZ>& moduli, unsigned long n){
    if(n == 1){
        dst = elements[0] % moduli[0];
    }
    ZZ a, p, A, P;
    a = elements[0];
    p = moduli[0];
    A = elements[1];
    P = moduli[1];
    int res = CRT(a,p,A,P);
    for(int i = 2; i < n; i++){
        res = CRT(a, p, elements[i], moduli[i]);
    }
    dst = a;
}

void reduceColumnMod(Mat<ZZ>& M, long c, long end, ZZ* d){
    ZZ q, t1;
    long n = M.NumRows();
    for(int r = c; r < end; r++){
        //M[r][r] %= d[r];
        rem(M[r][r], M[r][r], d[r]);
        if(IsZero(M[r][r])){
            M[r][r] = d[r];
        }
        //q = M[r][c-1] / M[r][r];
        div(q, M[r][c-1], M[r][r]);
        for(int k = r; k < n; k++){
            //M[k][c-1] -= q * M[k][r];
            //M[k][c-1] %= d[k];
            mul(t1, q, M[k][r]);
            sub(M[k][c-1],M[k][c-1], t1);
            rem(M[k][c-1], M[k][c-1], d[k]);
        }
        if(IsZero(M[c-1][c-1])){
            M[c-1][c-1] = d[c-1];
        }
    }
}


void generateZero(Mat<ZZ>& M, long row, long col){
    if(IsZero(M[row-1][col-1])){
        return;
    }
    long j = row;
    ZZ g, r, y, t1, t2, t3, b_i_save;
    XGCD(g, r, y, M[j-1][j-1], M[j-1][col-1]);
    ZZ bg = -M[j-1][col-1] / g;
    ZZ hg = M[j-1][j-1] / g;
    for(int i = j; i < M.NumRows() + 1; i++){
        b_i_save = ZZ(M[i-1][col-1]);
        //M[i-1][col-1] = (M(i, j) * bg + M[i-1][col-1] * hg);
        mul(t1, M[i-1][j-1], bg);
        mul(t2, M[i-1][col-1], hg);
        add(M[i-1][col-1], t1, t2);
        //M[i-1][j-1] = (r * M(i,j) + y * b_i_save);
        mul(t1, M[i-1][j-1], r);
        mul(t2, y, b_i_save);
        add(M[i-1][j-1], t1, t2);
    }
}

void generateZeroRowMod(Mat<ZZ>& M, long row, ZZ* d){
    long idx = M.NumCols() / 2 + row - 1;
    M.put(row-1, idx, d[row-1]);
    for(long col = row+1; col < idx + 2; col++){
        generateZero(M, row, col);
    }
    ZZ t;
    for(int c = row; c < idx + 2; c++){
        for(int r = row; r < M.NumRows() + 1; r++){
            rem(t, M[r-1][c-1], d[r-1]);
            M.put(r-1, c-1, t);
        }
    }
    if(IsZero(M[row-1][row-1])){
        M.put(row-1, row-1, d[row-1]);
    }
}

void hnfModD(Mat<ZZ>& dst, Mat<ZZ>& M, ZZ& det) {
    Mat<ZZ> B;
    long m = M.NumRows();
    B.SetDims(m, m+m);
    ZZ D[m];
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            B[i][j] = M[i][j];
        }
        B[i][m+i] = det;
        D[i] = det;
    }
    long n = m+m;//number of columns
    for(long i = 1; i < m+1; i++){
        generateZeroRowMod(B, i, D);
        for(long j = 1; j < i; j++){
            reduceColumnMod(B, j, i, D);
        }
        if( i < m ) {
            //D[i] = D[i - 1] / B[i - 1][i - 1];
            div(D[i], D[i-1], B[i-1][i-1]);
        }
    }
    dst.SetDims(m,m);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            dst[i][j] = B[i][j];
        }
    }


}



void HNFModD(Mat<ZZ>& dst, Mat<ZZ>& M, ZZ& det){
    int n = M.NumRows();
    Mat<ZZ> trans;
    ZZ tmp;
    transpose(trans, M);
    HNF(dst, M, det);
    //correction
    for(int i = 0; i < n-1; i++){
        for(int j = 0; j < n - i - 1; j++){
            tmp = dst[i][j];
            dst.put(i, j, dst[n - j - 1][n - i - 1]);
            dst.put(n - j - 1,n - i - 1,tmp);
        }
    }
}

void addRowToMatrix(Mat<ZZ>& m, Vec<ZZ>& v){
    int rows = m.NumRows();
    int cols = m.NumCols();
    if(cols == 0 && rows == 0){
        m.SetDims(1, v.length());
        for (int i = 0; i < v.length(); i++) {
            m[rows][i] = v[i];
        }
    }else if(cols != v.length()){
        cout << "addRowToMatrix: dimension missmatch" << endl;
    }else{
        m.SetDims(rows + 1, cols);
        for (int i = 0; i < cols; i++) {
            m[rows][i] = v[i];
        }
    }
}

void addColumnToMatrix(Mat<ZZ>& dst, Mat<ZZ>& m, Vec<ZZ>& v){
    int rows = m.NumRows();
    int cols = m.NumCols();
    if(rows != v.length()){
        cout << "addColumnTomatrix: dimension missmatch" << endl;
    }
    if(dst == m){
        cout << "addColumnTomatrix: dst and m must be different!" << endl;
    }
    dst.SetDims(rows, cols+1);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            dst[i][j] = m[i][j];
        }
        dst[i][cols] = v[i];
    }
}

void getPrincipalMinor(Mat<ZZ>& dst, Mat<ZZ>& a, unsigned long m) {
    dst.SetDims(m, m);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            dst[i][j] = a[i][j];
        }
    }
}

void get_a_t(Vec<ZZ>& v, Mat<ZZ>& m, long idx){
    v.SetLength(idx);
    for(int i = 0; i < idx; i++){
        v[i] = m[idx][i];
    }

}

void get_a_column(Vec<ZZ>& v, Mat<ZZ>& m, long idx){
    v.SetLength(idx);
    for(int i = 0; i < idx; i++){
        v[i] = m[i][idx-1];
    }
}

void get_e_i(Vec<ZZ>& v, long n, long idx){
    v.SetLength(n);
    v[idx] = 1;
}

ZZ getDiagProd(Mat<ZZ>& m){
    long r = m.NumRows();
    ZZ res = ZZ(1);
    for(int i = 0; i < r; i++){
        res *= m[i][i];
    }
    return res;
}

ZZ getMaxColumnLength(Mat<ZZ>& m){
    ZZ max = ZZ(-1);
    long cols = m.NumCols();
    long rows = m.NumRows();
    for(int j = 0; j < cols; j++){
        ZZ res = ZZ(0);
        for(int i = 0; i < rows; i++){
            res += (m[i][j] * m[i][j]);
        }
        ZZ norm = SqrRoot(res);
        if(norm > max){
            max = norm;
        }
    }
    return max;
}

ZZ getMaxValue(Mat<ZZ>& B){
    ZZ max = ZZ(-1);
    for(int i = 0; i < B.NumRows(); i++){
        for(int j = 0; j < B.NumCols(); j++){
            if(abs(B[i][j]) > max){
                max = B[i][j];
            }
        }
    }
    return ZZ(max);
}

void addRow(Vec<ZZ>& dst, Mat<ZZ>& B, Mat<ZZ>& H_B, Vec<ZZ>& a_t){
    long n = B.NumRows();
    ZZ D = abs(getDiagProd(H_B));
    ZZ product = ZZ(1);
    Vec<ZZ> chose_primes;
    /*ZZ max = ZZ(0);

    addRowToMatrix(B, a_t);
    for(int i = 0; i < n+1; i++)
        for(int j = 0; j < n; j++){
            if(B[i][j] > max){
                max = B[i][j];
            }
        }
    ZZ p = max;*/

    B.SetDims(n, n);
    ZZ p = ZZ((unsigned long)pow(2, 63));
    //ZZ M_ = getMaxColumnLength(B) + ZZ(1);
    //ZZ bound = power(M_, n);
    ZZ M;
    M = getMaxValue(B);
    for(int i = 0; i < a_t.length(); i++){
        if(abs(a_t[i]) > M){
            M = ZZ(a_t[i]);
        }
    }
    ZZ bound, t, t1;
    t = ZZ(n);
    t = power(t,n+1);
    bound = t * power(M, 2*n+1);
    while(product <= 2 * bound){
        p = NextPrime(p+1);
        if(!divide(D, p)){
            chose_primes.append(p);
            product *= p;
        }
    }
    Mat<ZZ> x_vectors;
    Mat<ZZ> transposed_H_B, B_T;
    transpose(B_T, B);
    transpose(transposed_H_B, H_B);
    for(int i = 0; i < chose_primes.length(); i++){
        // there is probably a faster implementation based on NTL for this -> change modulus too often e.g.
        ZZ_p::init(chose_primes[i]);
        Mat<ZZ_p> B_Copy, H_B_T_mod;
        conv(B_Copy, B_T);// convert B to a B mod p_i
        conv(H_B_T_mod, transposed_H_B);
        Vec<ZZ_p> x,y, a_t_copy;
        conv(a_t_copy, a_t);
        ZZ_p det;
        solve(det, B_Copy, y, a_t_copy);
        x = H_B_T_mod * y;
        Vec<ZZ> nx;
        conv(nx, x);
        addRowToMatrix(x_vectors, nx);
    }
    Vec<ZZ> x_res;
    for(int i = 0; i < x_vectors.NumCols(); i++){
        Vec<ZZ> coords;
        for(int j = 0; j < x_vectors.NumRows(); j++){
            coords.append(x_vectors[j][i]);
        }
        ZZ result;
        CRTOnArray(result, coords, chose_primes, chose_primes.length());
        x_res.append(result);
    }

    dst = x_res;

    chose_primes.kill();
    B_T.kill();
    transposed_H_B.kill();
    x_vectors.kill();
    x_res.kill();
    bound.kill();
    //M_.kill();
    D.kill();
    product.kill();
    //p.kill();
}

void addRowWithoutCRT(Vec<ZZ>& dst, Mat<ZZ>& B, Mat<ZZ>& H_B, Vec<ZZ>& a_t){
    RR::SetOutputPrecision(150);
    RR::SetPrecision(150);
    long n = B.NumRows();
    RR det;
    Vec<RR> y, x, a_tR;
    Mat<RR> BR, H_BR;
    BR.SetDims(n,n);
    H_BR.SetDims(n,n);
    a_tR.SetLength(n);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            conv(BR[i][j], B[i][j]);
            conv(H_BR[i][j], H_B[i][j]);

        }
        conv(a_tR[i], a_t[i]);
    }
    solve(det, y, BR, a_tR);
    x = y * H_BR;

    Vec<ZZ> nx;
    nx.SetLength(n);

    //conv(nx, x);
    for(int i = 0; i < n; i++){
        conv(nx[i],round(x[i]));
    }

    dst = nx;

    BR.kill();
    H_BR.kill();
    nx.kill();
}


void addColumn(Mat<ZZ>& dst, Mat<ZZ>& A, Vec<ZZ>& b){
    //cout << "A=\n" << A << endl;
    //cout << "b=\n" << b <<  endl;
    long n = A.NumRows();
    Mat<ZZ> H_all = Mat<ZZ>(A);
    Mat<ZZ> tmp;
    addColumnToMatrix(tmp, H_all, b);
    ZZ d_;
    Vec<ZZ> c_;
    c_.SetLength(n);
    if(tmp.NumRows() == tmp.NumCols()) {
        //cout << "tmp=\n" << tmp << endl;
        d_ = abs(determinant(tmp));
        //cout << "-> d_ = " << d_ << endl;
        c_[n-1] = ZZ(d_);
    }
    Vec<ZZ> m;
    m.SetLength(n);
    if(H_all.NumCols() < H_all.NumRows()){
        if(d_ == 0){
            dst = H_all;
            return;
        }
        addColumnToMatrix(tmp, H_all, c_);
        H_all.SetDims(0,0);
        H_all = tmp;
        m[n-1] = abs(d_);
    }else{
        m[n-1]= abs(determinant(H_all));
    }
    //cout << "c_=\n" << c_ << endl;
    //cout << "d_=\n" << d_ << endl;
    //cout << "m=\n" << m << endl;
    //cout << "H_all=\n" << H_all << endl;
    for(int i = n-1; i > 0; i--){
        m[i-1] = abs(m[i] * H_all(i, i));
    }
    for(int j = 1; j < n+1; j++){
        ZZ g, r, y;
        XGCD(g, r, y, H_all(j, j), b[j-1]);
        ZZ bg = -b[j-1] / g;
        ZZ hg = H_all(j,j) / g;
        for(int i = j; i < n+1; i++){
            ZZ b_i_save = ZZ(b[i-1]);
            b[i-1] = (H_all(i, j) * bg + b[i-1] * hg) % m[i-1];
            H_all(i, j) = (r * H_all(i,j) + y * b_i_save) % m[i-1];
        }
        if(H_all(j, j) == 0){
            H_all(j,j) = m[j-1];
        }
        for(int c = 1; c < j; c++){
            ZZ q = H_all(j, c) / H_all(j, j);
            for(int r = j; r < n+1; r++){
                H_all(r, c) = (H_all(r, c) - q * H_all(r, j)) % m[r-1];
            }
        }
        for(int k = j+1; k < n+1; k++){
            ZZ q = H_all(k, j) / H_all(k, k);
            for(int l = k; l < n+1; l++){
                H_all(l, j) = (H_all(l, j) - q * H_all(l, k)) % m[l-1];
            }
        }
    }
    dst = H_all;
}


void hnf(Mat<ZZ>& dst, Mat<ZZ>& B){
    if(B[0][0] < 0){
        for(int i = 0; i < B.NumRows(); i++){
            NTL::negate(B[i][0], B[i][0]);
        }
    }
    long n = B.NumCols();
    if(n == 1){
        dst = Mat<ZZ>(B);
    }
    Mat<ZZ> H;
    getPrincipalMinor(H, B, 1);
    for(int i = 2; i < n+1; i++){
        Mat<ZZ> A, newmatrix;
        newmatrix = Mat<ZZ>(H);
        getPrincipalMinor(A, B, i-1);
        Vec<ZZ> a_t, b, x;
        get_a_t(a_t, B, i-1);
        addRow(x, A,H,a_t);

        addRowToMatrix(newmatrix, x);

        get_a_column(b, B, i);
        addColumn(H, newmatrix, b);
    }
    dst = H;
}

void decomposeForHeuristic(Mat<ZZ>& B,Vec<ZZ>& b_t,Vec<ZZ>& c,Vec<ZZ>& d, ZZ& a_1, ZZ& a_2, Mat<ZZ>& M){
    long n = M.NumRows();
    a_1 = M[n-1][n-2];
    a_2 = M[n-1][n-1];
    c.SetLength(n-1);
    d.SetLength(n-1);
    B.SetDims(n-1,n-2);
    b_t.SetLength(n-2);
    for(int i = 0; i < n-1; i++){
        for(int j = 0; j < n-2; j++){
            B[i][j] = M[i][j];
            b_t[j] = M[n-1][j];
        }
        c[i] = M[i][n-2];
        d[i] = M[i][n-1];
    }
}

void hnf_heuristic(Mat<ZZ>& dst, Mat<ZZ>& A){
    Mat<ZZ> B, tmp, H;
    Vec<ZZ> c,d,b, nv, x;
    ZZ a1,a2, d1, d2, g, k, l;
    decomposeForHeuristic(B, b, c, d, a1, a2, A);
    cout << "computed decomposition\n";
    addColumnToMatrix(tmp, B, c);
    d1 = determinant(tmp);
    addColumnToMatrix(tmp, B, d);
    d2 = determinant(tmp);
    cout << "computed determinants" << endl;
    XGCD(g, k, l, d1, d2);
    nv = k*c+l*d;
    addColumnToMatrix(tmp, B, nv);
    cout << "start to compute HNFModD\n";
    HNFModD(H, tmp, g);
    //hnfModD(H, tmp, g);
    cout << "done\n";
    b.append(k*a1+l*a2);
    //cout << "start addRow\n";




    //addRow(x, tmp, H, b);
    addRowWithoutCRT(x, tmp, H, b);

    addRowToMatrix(H, x);

    c.append(a1);
    addColumn(tmp, H, c);

    d.append(a2);
    addColumn(dst, tmp, d);




}


