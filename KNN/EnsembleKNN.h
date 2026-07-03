#pragma once
#include"KNN.h"
#include<vector>
#include<iostream>
#include<random>
#include<algorithm>
using namespace std;

// 集成式 KNN (Ensemble KNN)，利用 Bagging 的概念訓練多個 KNN 分類器
class ENKNN
{
private:
    const unsigned int numOfKNN;               // 要建構的 KNN 子模型數量
    const unsigned int KSize;                  // 子模型從訓練集中隨機抽樣 (Bagging) 的樣本數
    vector<KNN> KNNvec;                        // 儲存所有的子模型 (KNN 實體)
    vector<vector<unsigned int>> subIndices;   // 儲存每個子模型被分配到的隨機訓練資料索引

public:
    ENKNN(unsigned int n, unsigned int K);     // n=子模型數, K=KNN的K值
    
    // 透過隨機抽樣 (抽後放回) 來替所有的子模型準備資料索引
    void train(Data &trainedData, unsigned int minSize, unsigned int maxSize);
    
    // 統合所有 KNN 子模型的預測，依據 Gini Impurity (信心水準) 進行加權投票
    double classify(const vector<double> &input, Data &trainedData) const;
    
    // 對測試集進行評估，並將準確率寫入檔案中
    void showAccuracy(const string Path, Data &trainedData, ofstream &wfile, vector<double> &avg, unsigned int &i) const;
};