#include"KNN.h"
#include<cmath>

// 建構子：設定最近鄰居數 K
KNN::KNN(unsigned int setK):
    K(setK)
{
    //nothing
}

// 計算兩筆資料 (x, y) 之間的特徵距離 (曼哈頓距離 L1)
double KNN::distance(const vector<double> &x, const vector<double> &y, DataFeature& feature) const
{
    // 防呆：確認兩個向量長度一致
    if(x.size() != y.size())
    {
        cout<<"In KNN::distance, in put size uneven\n";
        for(const double &f : x)
        {
            cout<<f<<' ';
        }cout<<endl;
        for(const double &f : y)
        {
            cout<<f<<' ';
        }cout<<endl;
        exit(1);
    }

    unsigned int size = x.size();
    double distance = 0.0;
    switch(feature.getEncodedType())
    {
        case ONE_HOT_ENCODING:
        {
            for(const VariableFeature& curFeature: feature.getFeatures())
            {
                if(curFeature.type == BOOLLABEL)   // 目標標籤不列入距離計算
                    continue;
                unsigned int offset = curFeature.vectorOffset;
                distance +=fabs(x[offset] - y[offset]);
            }
            break;
        }
        case LABEL_ENCODING:
        {
            for(const VariableFeature& curFeature: feature.getFeatures())
            {
                unsigned int offset = curFeature.vectorOffset;
                switch(curFeature.type)
                {
                    case NUMERICAL :
                    {
                        // 連續數值的距離 (經過 Z-score 標準化後的差異)
                        distance +=fabs(x[offset] - y[offset]);
                        break;
                    }
                    case BOOL:
                    case CATEGORICAL :
                    {
                        // 若為未知的遺失值 (< 0)，直接給予最大距離 1.0
                        if (x[offset] < 0 || y[offset] < 0)
                        {
                            distance += 1.0;
                        } 
                        else
                        {
                            // 類別特徵使用 Target Encoding (信心水準) 來算距離
                            double confX = curFeature.confidence[(unsigned int)x[offset]];
                            double confY = curFeature.confidence[(unsigned int)y[offset]];
                            // 在這裡，可以加上特徵權重來平衡數值與類別的差距
                            distance += fabs(confX - confY);
                        }
                        break;
                    }
                }
            }
            break;
        }
    }
    return distance;
}

// 在整個訓練集中尋找與 comparedData 最接近的 K 個鄰居
void KNN::foundNeighbors(
    const vector<double> &comparedData, 
    vector<pair<unsigned int, double>> &neighbors, 
    Data &trainedData) const
{
    // 定義 Priority Queue 比較函式 (Max-Heap：距離最遠的放最上面)
    auto compare = [&](const pair<unsigned int, double> &data1, const pair<unsigned int, double> &data2)
    {
        return data1.second < data2.second;
    };
    // 建立 Max-Heap (存放 {索引, 距離})
    priority_queue<pair<unsigned int, double>, vector<pair<unsigned int, double>>, decltype(compare)>  pq(compare);
    const vector<vector<double>>& localData = trainedData.getData();
    
    // 計算所有資料與目標的距離並加入 pq
    for(unsigned int i=0; i<localData.size(); i++)
    {
        pq.push({i, distance(comparedData, localData[i], trainedData.getDataFeature())});
        // 保持 pq 裡面最多只有 K 個元素 (踢掉距離最遠的)
        if(pq.size() > K)
            pq.pop();
    }
    // 將最後留在 pq 裡面的 K 個元素搬到 neighbors 陣列
    while(!pq.empty())
    {
        pair<unsigned int, double> p = pq.top();
        neighbors.push_back(p);
        pq.pop();
    }
}

// 在指定的子集 (index) 中尋找最接近的 K 個鄰居 (Bagging 使用)
void KNN::foundNeighbors(
    const vector<double> &comparedData, 
    vector<pair<unsigned int, double>> &neighbors, 
    Data &trainedData,
    const vector<unsigned int> &index) const
{
    auto compare = [&](const pair<unsigned int, double> &data1, const pair<unsigned int, double> &data2)
    {
        return data1.second < data2.second;
    };
    priority_queue<pair<unsigned int, double>, vector<pair<unsigned int, double>>, decltype(compare)>  pq(compare);
    const vector<vector<double>>& localData = trainedData.getData();
    
    for(unsigned int i=0; i<index.size(); i++)
    {
        pq.push({index[i], distance(comparedData, localData[index[i]], trainedData.getDataFeature())});
        if(pq.size() > K)
            pq.pop();
    }
    while(!pq.empty())
    {
        pair<unsigned int, double> p = pq.top();
        neighbors.push_back(p);
        pq.pop();
    }
}

// 對輸入的資料進行預測分類
double KNN::classify(const vector<double> &input, Data &trainedData) const
{
    if(!trainedData.format(input))
    {
        cout<<"In KNN::classify, input is invalid format."<<endl;
        for(const double &f: input)
            cout<<f<<' ';
        cout<<endl;
        exit(1);
    }

    vector<pair<unsigned int, double>> neighbors;
    neighbors.reserve(K);
    foundNeighbors(input, neighbors, trainedData);
    double approveDist = 0;
    
    VariableFeature labelFeature = trainedData.getDataFeature().getLable();
    const vector<vector<double>> &localData = trainedData.getData();
    
    switch(labelFeature.type)
    {
        case BOOLLABEL:
        {
            // 距離加權投票：距離越近權重越高 (1 / (1+d))
            for(pair<unsigned int, double> p: neighbors)
            {
                if(localData[p.first][labelFeature.vectorOffset] == 0.0)
                    approveDist += 1.0/(1.0 + p.second);
                if(localData[p.first][labelFeature.vectorOffset] == 1.0)
                    approveDist -= 1.0/(1.0 + p.second);
            }
            // 投票結算：若 approveDist 大於 0 代表 0.0 的票數多
            if(approveDist > 0 )
                return 0.0;
            else    
                return 1.0;
            break;
        }
        case NUMERICALLABEL:
        {
            // 迴歸預測：使用距離加權的數值平均
            double totalDist = 0.0;
            for(pair<unsigned int, double> p: neighbors)
            {
                approveDist += 1.0/(1.0 + p.second) * localData[p.first][labelFeature.vectorOffset];
                totalDist += 1.0/(1.0 + p.second);
            }
            return round(approveDist / totalDist);
            break;
        }
    }
    return -1.0;
}

// 分類並同步計算該鄰域的 Gini Impurity (供 Ensemble 加權使用)
double KNN::classify(const vector<double> &input, Data &trainedData, const vector<unsigned int> &index, double &giniImpurity) const
{
    if(!trainedData.format(input))
    {
        cout<<"In KNN::classify, input is invalid format."<<endl;
        for(const double &f: input)
            cout<<f<<' ';
        cout<<endl;
        exit(1);
    }

    vector<pair<unsigned int, double>> neighbors;
    neighbors.reserve(K);
    foundNeighbors(input, neighbors, trainedData, index);
    
    double approveDist = 0.0;
    unsigned int approveCount = 0;
    unsigned int disApproveCount = 0;
    VariableFeature labelFeature = trainedData.getDataFeature().getLable();
    const vector<vector<double>> &localData = trainedData.getData();

    switch(labelFeature.type)
    {
        case BOOLLABEL:
        {
            for(pair<unsigned int, double> p: neighbors)
            {
                if(localData[p.first][labelFeature.vectorOffset] == 0.0)
                    approveDist += 1.0/(1.0 + p.second);
                if(localData[p.first][labelFeature.vectorOffset] == 1.0)
                {
                    approveDist -= 1.0/(1.0 + p.second);
                    approveCount++;
                }
            }
            // 計算鄰居範圍內的 Gini Impurity，代表模型對這次預測的「信心度」
            disApproveCount = neighbors.size() - approveCount;
            double approveProb = (double)approveCount/neighbors.size();
            double disApprovProb = (double)disApproveCount/neighbors.size();
            giniImpurity = 1.0 - (approveProb * approveProb + disApprovProb * disApprovProb);
            
            if(approveDist > 0 )
                return 0.0;
            else    
                return 1.0;
        }
        case NUMERICALLABEL:
        {
            double totalDist = 0.0;
            approveDist = 0.0;
            vector<double> uniqueLabels;
            vector<int> counts;
            
            for(pair<unsigned int, double> p: neighbors)
            {
                double labelVal = localData[p.first][labelFeature.vectorOffset];
                approveDist += 1.0/(1.0 + p.second) * labelVal;
                totalDist += 1.0/(1.0 + p.second);
                
                // 統計不重複數值的出現次數，以計算不純度
                bool found = false;
                for(unsigned int i = 0; i < uniqueLabels.size(); i++) {
                    if(uniqueLabels[i] == labelVal) {
                        counts[i]++;
                        found = true;
                        break;
                    }
                }
                if(!found) {
                    uniqueLabels.push_back(labelVal);
                    counts.push_back(1);
                }
            }
            
            double impurity = 1.0;
            for(int count : counts)
            {
                double prob = (double)count / neighbors.size();
                impurity -= prob * prob;
            }
            giniImpurity = impurity;

            if (totalDist == 0.0) return 0.0;
            return round(approveDist / totalDist);
        }
    }
    return 0.0;
}

// 驗證測試集準確率
void KNN::showAccuracy(const string Path, Data &trainedData, ofstream &wfile, vector<double> &avg, unsigned int &i) const
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

    // 逐行讀取測試集資料
    while(!file.eof())
    {
        singleData.clear();
        singleData = trainedData.trancform(file);
        if(singleData.empty())
            break;
        testCount++;
        
        // 分類並比較結果
        double result = classify(singleData, trainedData);
        if(singleData[labelFeature.vectorOffset] == result)
        {
            correctCount++;
        }
    }

    // 輸出正確數與準確率
    wfile<<"Correct count : "<<correctCount<<endl;
    wfile<<"Accuracy : "<<(double)correctCount / testCount<<endl;
    avg[i] += (double)correctCount / testCount;
}
