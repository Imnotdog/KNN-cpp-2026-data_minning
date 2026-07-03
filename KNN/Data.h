#pragma once
#include<cmath>
#include"Feature.h"
using namespace std;

// 負責管理與處理原始資料的類別
class Data 
{
private:
    bool featureExist;             // 是否已經載入特徵定義
    DataFeature dataFeature;       // 資料的所有特徵定義與統計資訊
    vector<vector<double>> data;   // 儲存實際轉換過後的訓練資料表
    
    // 從檔案中讀取資料，並轉換成數值存入 vector 中
    void readData(const string &Path);
    // 對數值特徵做 Z-score 標準化，對類別特徵算目標存活率
    void preProcess();

public:
    // 透過已有的特徵定義來載入資料
    Data(const string dataPath, const DataFeature &features);
    // 透過提供特徵描述檔 (attrib) 來載入並建構資料與特徵定義
    Data(const string dataPath, const string typePath, const vector<Attrib> &attrib, const ENCODING setEncodeType = LABEL_ENCODING);
    
    // 讀取檔案的一行，將其轉換成經過處理的特徵向量 (用於測試集)
    vector<double> trancform(ifstream &file);
    
    // 取得資料特徵定義
    DataFeature& getDataFeature();
    // 取得內部已處理好的資料矩陣
    const vector<vector<double>>& getData() const;
    
    // 取得單筆資料轉換為 vector 後的長度
    unsigned int getSingleDataSize() const;
    // 檢查輸入的資料是否符合我們定義的特徵格式 (防呆機制)
    bool format(const vector<double> &vec);
};
