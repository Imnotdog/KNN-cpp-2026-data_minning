#pragma once
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<random>
#include<numeric>
#include<algorithm>
#include<vector>

using namespace std;

// 讀取檔案內容並存入 vector
void read(string path, vector<string> &data);
// 將 vector 內的資料寫入檔案
void write(string path, vector<string> data);
// 將測試集與訓練集合併、洗牌後，重新切割並覆寫檔案
void shullfeData(string TESTPATH, string TRAINPATH, int TRAINNUM);
