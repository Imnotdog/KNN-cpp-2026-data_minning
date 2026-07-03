#include"EnsembleKNN.h"

// 建構子：初始化 n 個 KNN 子模型
ENKNN::ENKNN(unsigned int n, unsigned int K) : numOfKNN(n), KSize(K)
{
    // 建立 n 個 K 值的 KNN 實體
    KNNvec = vector<KNN>(n, KNN(K));
}

// 訓練階段：透過隨機抽樣 (無後放回的隨機子集) 為每個 KNN 分配不同的訓練資料
void ENKNN::train(Data &trainedData, unsigned int minSize, unsigned int maxSize)
{
    unsigned int totalDataSize = trainedData.getData().size();
    
    // 建立一個完整的索引陣列 (0 到 totalDataSize-1)
    vector<unsigned int> index(totalDataSize);
    for(unsigned int i = 0; i < totalDataSize; i++)
    {
        index[i] = i;
    }
    
    // 亂數產生器
    random_device rd;
    mt19937 gen(rd());
    // 設定每個子模型抽樣數量的亂數範圍
    uniform_int_distribution<unsigned int> sizeDist(minSize, maxSize);

    subIndices.clear();
    // 為每一個 KNN 模型抽樣
    for(unsigned int i = 0; i < numOfKNN; i++)
    {
        // 打亂索引順序
        shuffle(index.begin(), index.end(), gen);
        
        // 決定這次要抽幾筆資料
        unsigned int currentSize = sizeDist(gen);
        if (currentSize > totalDataSize) currentSize = totalDataSize;
        
        // 擷取前 currentSize 筆索引作為這個子模型的專屬訓練集 (Bagging)
        vector<unsigned int> sub(index.begin(), index.begin() + currentSize);
        subIndices.push_back(sub);
    }
}

// 預測階段：集合所有 KNN 子模型的預測結果進行加權投票
double ENKNN::classify(const vector<double> &input, Data &trainedData) const
{
    double totalDiatance = 0.0;
    double approveDistance = 0.0;
    
    for(unsigned int i = 0; i < numOfKNN; i++)
    {
        double giniImpurity = 0.0;
        // 讓第 i 個 KNN 依據它自己的子資料集進行預測，並同時計算 Gini Impurity
        double approve = KNNvec[i].classify(input, trainedData, subIndices[i], giniImpurity);
        
        double confidence = 1.0; // 預設信心水準
        
        switch(trainedData.getDataFeature().getLable().type)
        {
            case BOOLLABEL:
            {
                // 對於二元分類：將 Gini (0~0.5) 轉換為信心權重 (1.0~0.0)
                confidence = 1.0 - 2.0 * giniImpurity;
                if(confidence < 0.0) confidence = 0.0;
                
                // 若模型預測 1.0，則投正票 (+confidence)；預測 0.0 投負票 (-confidence)
                approveDistance += (approve == 1.0 ? 1 : -1) * confidence;
                break;
            }
            case NUMERICALLABEL:
            {
                // 對於數值迴歸：Gini 轉換為信心權重
                confidence = 1.0 - giniImpurity;
                if(confidence < 0.0) confidence = 0.0;
                
                // 加權平均累加
                approveDistance += approve * confidence;
                totalDiatance += confidence;
                break;
            }
        }
    }
    
    // 結算所有模型的投票結果
    switch(trainedData.getDataFeature().getLable().type)
    {
        case BOOLLABEL:
        {
            // 如果正票多於負票，代表多數且高信心的模型認為是 1.0
            return approveDistance > 0.0 ;
        }
        case NUMERICALLABEL:
        {
            // 防呆：避免除以 0
            if (totalDiatance == 0.0f) return 0.0f;
            return round(approveDistance / totalDiatance);
        }
    }
    return 0.0; // 預設回傳
}

// 評估 Ensemble KNN 的準確率
void ENKNN::showAccuracy(const string Path, Data &trainedData, ofstream &wfile, vector<double> &avg, unsigned int &i) const
{
    ifstream file(Path);
    if(file.fail())
    {
        cout<<"In KNN::showAccuracy, can't open "<<Path<<endl;
        exit(1);
    }
    vector<double> singleData;
    unsigned int correctCount = 0;
    unsigned int testCount = 0;
    VariableFeature labelFeature = trainedData.getDataFeature().getLable();
    
    // 逐筆讀取測試集
    while(!file.eof())
    {
        singleData.clear();
        singleData = trainedData.trancform(file);
        if(singleData.empty())
            break;
        testCount++;
        
        // 取得預測結果並比對真實標籤
        double result = classify(singleData, trainedData);
        if(singleData[labelFeature.vectorOffset] == result)
        {
            correctCount++;
        }
    }
    wfile<<"Correct count : "<<correctCount<<endl;
    wfile<<"Accuracy : "<<(double)correctCount / testCount<<endl;
    avg[i] += (double)correctCount / testCount;
}