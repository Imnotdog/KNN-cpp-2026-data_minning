#pragma once
#include"Data.h"
#include<iostream>
#include<string>
#include<vector>
#include<queue>

using namespace std;

// KNN 模型類別，用於計算距離與分類
class KNN
{
private:
    const unsigned int K; // 設定最近鄰居的數量 K
    
    // 計算兩筆資料 (x, y) 之間的特徵距離
    double distance(const vector<double> &x, const vector<double> &y, DataFeature& feature) const;
    
    // 從 trainedData 尋找與 comparedData 距離最近的 K 個鄰居 (一般訓練集)
    void foundNeighbors(const vector<double> &comparedData, 
        vector<pair<unsigned int, double>> &neighbors, 
        Data &trainedData) const;

    // 從 trainedData 尋找與 comparedData 距離最近的 K 個鄰居 (搭配子索引 - Bagging 用)
    void foundNeighbors(const vector<double> &comparedData, 
        vector<pair<unsigned int, double>> &neighbors, 
        Data &trainedData, 
        const vector<unsigned int> &index) const;
public:
    KNN(unsigned int K); // 建構子，初始化 K 值
    
    // 預測輸入資料 input 的類別
    double classify(const vector<double> &input, Data &trainedData) const;
    
    // 預測類別，並同時計算該預測範圍內的 Gini 不純度 (Ensemble 使用)
    double classify(const vector<double> &input, Data &trainedData, const vector<unsigned int> &index, double &accuracy) const;
    
    // 給定測試集檔案路徑，輸出預測正確率到 wfile
    void showAccuracy(const string Path, Data &trainedData, ofstream &wfile, vector<double> &avg, unsigned int &i) const;
};