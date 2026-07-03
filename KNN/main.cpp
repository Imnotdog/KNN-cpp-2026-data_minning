#include"Data.h"
#include"KNN.h"
#include"EnsembleKNN.h"
#include"shuffleData.h"
#include<iostream>
#include<string>
#include<vector>

// 定義資料集相關檔案路徑
#define TESTPATH "TitanicTest.csv"
#define TRAININGPATH "TitanicTraining.csv"
#define ATTRIBPATH "TitanicArttib.txt"
using namespace std;

// 儲存每個參數設定在多次交叉驗證 (Cross-Validation) 中的平均準確率
vector<double> avg;

int main(void)
{
    avg.resize(14+30, 0); // 這裡預先配置足夠的空間來儲存各種 K 的準確率加總
    
    // 定義鐵達尼號 (Titanic) 資料集的特徵型態與名稱
    vector<Attrib> attribType = 
    {
        {"PassengerId", IGNORE},      // 忽略 ID 欄位
        {"Survived", BOOLLABEL},      // 目標預測變數：存活與否 (二元分類)
        {"Pclass", NUMERICAL},        // 艙等
        {"Name", IGNORE},             // 忽略姓名
        {"Sex", BOOL},                // 性別
        {"Age", NUMERICAL},           // 年紀
        {"SibSp", NUMERICAL},         // 兄弟姊妹數
        {"Parch", NUMERICAL},         // 父母小孩數
        {"Ticket", IGNORE},           // 忽略票根編號
        {"Fare", NUMERICAL},          // 票價
        {"Cabin", IGNORE},            // 忽略艙位號碼 (遺失值過多)
        {"Embarked", CATEGORICAL}     // 登船港口 (多分類)
    };

    /* 
    // 下面是被註解掉的紅酒 (Wine) 資料集特徵定義
    vector<Attrib> attribType = 
    {
        {"fixed acidity", NUMERICAL},
        {"volatile acidity", NUMERICAL},
        {"citric acid", NUMERICAL},
        {"residual sugar", NUMERICAL},
        {"chlorides", NUMERICAL},
        {"free sulfur dioxide", NUMERICAL},
        {"total sulfur dioxide", NUMERICAL},
        {"density", NUMERICAL},
        {"pH", NUMERICAL},
        {"sulphates", NUMERICAL},
        {"alcohol", NUMERICAL},
        {"quality", NUMERICALLABEL},
        {"Id", IGNORE}
    };*//*
    vector<Attrib> attribType = 
    {
        {"age", NUMERICAL},
        {"job", CATEGORICAL},
        {"marital", CATEGORICAL},
        {"education", CATEGORICAL},
        {"default", BOOL},
        {"balance", NUMERICAL},
        {"housing", BOOL},
        {"loan", BOOL},
        {"contact", CATEGORICAL},
        {"day", NUMERICAL},
        {"month", CATEGORICAL},
        {"duration", NUMERICAL},
        {"campaign", NUMERICAL},
        {"pdays", NUMERICAL},
        {"previous", NUMERICAL},
        {"poutcome", CATEGORICAL},
        {"deposit", BOOLLABEL}
    };*/
    
    ofstream wfile ("record.txt"); // 寫入結果的檔案

    // 重複進行 20 次實驗，類似 Monte Carlo Cross-Validation
    for(int count = 0; count< 20 ;count++)
    {
        // 第一步：將所有資料混合並重新切出 713 筆作為訓練集，剩下作測試集
        shullfeData(TESTPATH, TRAININGPATH, 713);
        unsigned int j = 0;
        
        // 讀取並前處理訓練資料 (建立 Data 實體)
        Data data1(TRAININGPATH, ATTRIBPATH, attribType, LABEL_ENCODING);
    
        cout<<"count : "<<count<<endl;
        
        // 測試純 KNN 模型在各種 K 值下的準確率
        vector<unsigned int> sks = {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29};
        for(int k : sks)
        {
            wfile<<"k = "<<k<<endl;
            KNN knn(k);
            // 分類測試集並將結果寫入
            knn.showAccuracy(TESTPATH, data1, wfile, avg, j);
            j++;
        }
        
        
        
        // 測試 Ensemble KNN (整合學習模型) 的準確率 (已被註解)
        vector<unsigned int> num = {50,75,100,125,150}; // 子模型數量
        vector<unsigned int> ks = {13,15,17,19,21,23};         // 每個子模型的 K 值
       
        for(unsigned int n: num)
        {
            for(int k : ks)
            {
                wfile<<"n = "<<n<<", k = "<<k<<endl;
                ENKNN ENknn(n, k);          
                // 從 713 筆中，每次隨機抽取 350 到 550 筆來訓練一個子模型
                ENknn.train(data1, 350, 550);  //titanic
                ENknn.train(data1, 458, 733);  //wine
                ENknn.train(data1, 4465, 7144);  //bank
                ENknn.showAccuracy(TESTPATH, data1, wfile, avg, j);
                j++;
            }
        }
        
        // 輸出當前回合的累積平均準確率
        for(int i=0; i<avg.size(); i++)
        {
            wfile<<avg[i] / (count+1)<<endl;
        }
    }
    
    return 0;
}