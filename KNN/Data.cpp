#include"Data.h"

// 讀取原始資料表 (csv 等格式)，將文字轉換為初始特徵矩陣
void Data::readData(const string &Path)
{
    ifstream file(Path);
    if(file.fail())
    {
        cout<<"can't open "<<Path<<endl;
        exit(1);
    }
    char line[1000];
    string token;
    stringstream ss;
    // 取得資料的特徵定義與編碼方式
    vector<VariableFeature> localDataFeature = dataFeature.getFeatures();
    ENCODING encodedType = dataFeature.getEncodedType();
    
    while(!file.eof())
    {
        file.getline(line, 1000);
        ss.str("");
        ss.clear();
        ss.str(line);
        // 如果讀到自定義的 stop 標記，則提早結束讀取
        if(ss.str() == "stop")
        {
            break;
        }

        // 宣告一個固定長度的 vector 存放轉換後的一筆資料
        vector<double> curData(dataFeature.getSingleDataSize(), 0.0);
        
        bool miss = false; // 紀錄這筆資料是否有遺漏或無效的特徵
        for(unsigned int i=0; i<localDataFeature.size(); i++)
        {
            getline(ss, token, ',');
            // 特例處理：Name 欄位內含逗號，因此需多讀取一次以略過
            if(localDataFeature[i].name == "Name")
            {
                getline(ss, token, ',');
            }
            switch(localDataFeature[i].type)
            {
                case CATEGORICAL:
                {
                    // 若是未見過的類別值，視為無效資料並丟棄整筆
                    if(localDataFeature[i].index.find(token) == localDataFeature[i].index.end())
                    {
                        cout<<ss.str()<<endl;
                        cout<<"missing key, key = "<<token<<endl;
                        cout<<"ignored"<<endl<<endl;
                        miss = true;
                        break;
                    }
                    // 根據編碼方式轉換為數值
                    if(encodedType == ONE_HOT_ENCODING)
                    {
                        curData[localDataFeature[i].vectorOffset + localDataFeature[i].index[token]] = 1;
                    } 
                    else
                    {
                        curData[localDataFeature[i].vectorOffset] = localDataFeature[i].index[token];
                    }
                    break;
                }
                case NUMERICALLABEL:
                case NUMERICAL:
                {
                    // 連續數值直接由字串轉成 double
                    curData[localDataFeature[i].vectorOffset] = atof(token.c_str());
                    break;
                }
                case IGNORE:
                {
                    // 略過不處理
                    break;
                }
                case BOOL:
                case BOOLLABEL:
                {
                    // 布林特徵處理與 CATEGORICAL 類似
                    if(localDataFeature[i].index.find(token) == localDataFeature[i].index.end())
                    {
                        cout<<"missing key, key = "<<token<<endl;
                        cout<<"ignored"<<endl<<endl;
                        miss= true;
                        break;
                    }
                    curData[localDataFeature[i].vectorOffset] = localDataFeature[i].index[token];
                    break;
                }
            }
        }
        // 只有完整沒有遺失的資料才會加入 data 中
        if(!miss)
            data.push_back(curData);
    }
    file.close();

    // 更新設定 (如果有需要)
    dataFeature.setFeatures(localDataFeature);
}   

// 前處理階段：計算數值特徵的平均與變異數，以及計算類別特徵的目標機率 (Target Encoding)
void Data::preProcess()
{
    // 取得特徵定義的參照，以便直接修改內部的統計量
    vector<VariableFeature> &localDataFeature = dataFeature.getFeatures();
    VariableFeature labelFeature = dataFeature.getLable();
    
    // 第一階段：掃描所有資料以累加數值與頻率
    for(vector<double> &singleData : data)
    {
        for(VariableFeature &curFeature : localDataFeature)
        {
            switch(curFeature.type)
            {
                case NUMERICALLABEL:
                case NUMERICAL: 
                {
                    // 計算總和與平方和，之後用以計算變異數
                    double f = singleData[curFeature.vectorOffset];
                    curFeature.mean += f;
                    curFeature.variance += f*f;
                    break;
                }
                case BOOL:
                case BOOLLABEL:
                {
                    // 紀錄每個布林類別 (0/1) 出現的次數
                    curFeature.frequency[(unsigned int)singleData[curFeature.vectorOffset]]++;
                    // 如果該筆資料的 label 為 1.0 (正樣本)，則增加 approveFrequency
                    if(singleData[labelFeature.vectorOffset] == 1.0f)
                    {
                        curFeature.approveFrequency[(unsigned int)singleData[curFeature.vectorOffset]]++;
                    }
                    break;
                }
                case CATEGORICAL:
                {
                    // 根據編碼方式找出類別對應的索引，累加 frequency 與 approveFrequency
                    switch(dataFeature.getEncodedType())
                    {
                        case ONE_HOT_ENCODING:
                        {
                            int symbol =0;
                            for(symbol=0; symbol<curFeature.size; symbol++)
                            {
                                if(singleData[curFeature.vectorOffset + symbol] == 1.0f)
                                {
                                    break;
                                }
                            }
                            curFeature.frequency[symbol]++;
                            if(singleData[labelFeature.vectorOffset] == 1.0f)
                            {
                                curFeature.approveFrequency[(unsigned int)singleData[curFeature.vectorOffset]]++;
                            }
                            break;
                        }
                        case LABEL_ENCODING:
                        {
                            curFeature.frequency[(unsigned int)singleData[curFeature.vectorOffset]]++;
                            switch(dataFeature.getLable().type)
                            {
                                case NUMERICALLABEL:
                                {
                                    curFeature.approveFrequency[(unsigned int)singleData[curFeature.vectorOffset]]+= singleData[curFeature.vectorOffset];
                                    break;
                                }
                                case BOOLLABEL:
                                {
                                if(singleData[labelFeature.vectorOffset] == 1.0f)
                                {
                                    curFeature.approveFrequency[(unsigned int)singleData[curFeature.vectorOffset]]++;
                                }
                                    break;
                                }
                            }
                            break;
                        }
                    }
                    break;
                }
                
            }
        }  
    }

    // 第二階段：將累積的數值轉換為實際的平均與變異數
    for(VariableFeature &curFeature : localDataFeature)
    {
        if(curFeature.type == NUMERICAL || curFeature.type == NUMERICALLABEL)
        {
            curFeature.mean /= data.size(); // E[X]
            curFeature.variance /= data.size(); // E[X^2]
            curFeature.variance -= curFeature.mean * curFeature.mean; // Var(X) = E[X^2] - E[X]^2
            // 防呆機制：如果變異數趨近於 0，強制設為 1.0 避免除以 0 導致發散
            if(curFeature.variance < 0.0000001) curFeature.variance = 1.0;
        }
    }
    
    // 第三階段：對每一筆資料執行標準化 (Z-score) 或目標機率轉換
    for(vector<double> &singleData : data)
    {
        for(VariableFeature &curFeature : localDataFeature)
        {
            int offset = curFeature.vectorOffset;
            switch(curFeature.type)
            {
                case NUMERICAL:
                {
                    // Z-score 標準化公式： (X - Mean) / StdDev
                    for(int j=0; j < curFeature.size; j++)
                    {
                        singleData[offset] = (singleData[offset] - curFeature.mean) / sqrt(curFeature.variance);
                    }
                    break;
                }
                case BOOL:
                case CATEGORICAL:
                {
                    // 目標編碼：計算該類別的目標存活率 (approveFrequency / frequency)
                    if(curFeature.frequency[(unsigned int)singleData[offset]] > 0)
                        curFeature.confidence[(unsigned int)singleData[offset]] = (double)curFeature.approveFrequency[(unsigned int)singleData[offset]] / 
                            curFeature.frequency[(unsigned int)singleData[offset]];
                    else
                    {
                        curFeature.confidence[(unsigned int)singleData[offset]] = 0.0f;
                    }
                }
                
            }
        }  
    }
    
}


// 建構子：從特徵描述檔初始化，然後讀取資料與前處理
Data::Data(const string dataPath, const string typePath, const vector<Attrib> &attrib, const ENCODING setEncodeType):
    dataFeature(typePath, attrib, setEncodeType)
{
    data.clear();
    featureExist = true;
    cout<<"Start reading data\n"; 
    readData(dataPath);
    cout<<"Start preprocessing.\n";
    preProcess();
}

// 建構子：直接傳入已有的特徵定義來讀取資料
Data::Data(const string dataPath, const DataFeature &features):
    featureExist(true),
    dataFeature(features)
{

    cout<<"Start reading data\n"; 
    readData(dataPath);
    cout<<"Start preprocessing.\n";
    preProcess();
}

// 讀取檔案的一行，轉換並輸出為特徵 vector (通常用於預測階段或處理 Test Set)
vector<double> Data::trancform(ifstream &file) 
{
    if(file.fail())
    {
        cout<<"In Data::transfrom, can't open file"<<endl;
        exit(1);
    }
    vector<double> result;
    if(file.eof())
        return result;
    char line[1000];
    string token;
    stringstream ss;
    const vector<VariableFeature> localDataFeature = dataFeature.getFeatures();
    ENCODING encodedType = dataFeature.getEncodedType();
    
    if(!file.eof())
    {
        file.getline(line, 1000);
        ss.str("");
        ss.clear();
        ss.str(line);
        // 遇到 stop 就結束
        if(ss.str() == "stop")
        {
            return result;
        }

        result = vector<double>(dataFeature.getSingleDataSize(), 0.0);
        
        for(unsigned int i=0; i<localDataFeature.size(); i++)
        {
            getline(ss, token, ',');
            // 處理 Name 特殊欄位的逗號問題
            if(localDataFeature[i].name == "Name")
            {
                getline(ss, token, ',');
            }
            switch(localDataFeature[i].type)
            {
                case CATEGORICAL:
                {
                    auto it = localDataFeature[i].index.find(token);
                    // 未知的類別標記為 -1.0
                    if(it == localDataFeature[i].index.end())
                    {
                        result[localDataFeature[i].vectorOffset] = -1.0f;
                        break;
                    }
                    if(encodedType == ONE_HOT_ENCODING) 
                        result[localDataFeature[i].vectorOffset + it->second] = 1;
                    else
                        result[localDataFeature[i].vectorOffset] = it->second;
                    break;
                }
                case NUMERICALLABEL:
                {
                    result[localDataFeature[i].vectorOffset] = atof(token.c_str());
                    break;
                }
                case NUMERICAL:
                {
                    // 將新的數值依照訓練好的 mean 和 variance 進行 Z-score 標準化
                    double f = atof(token.c_str());
                    result[localDataFeature[i].vectorOffset] = (f - localDataFeature[i].mean)/sqrt(localDataFeature[i].variance);
                    break;
                }
                case IGNORE:
                {
                    break;
                }
                case BOOL:
                case BOOLLABEL:
                {
                    auto it = localDataFeature[i].index.find(token);
                    if(it == localDataFeature[i].index.end())
                    {
                        result[localDataFeature[i].vectorOffset] = -1.0f;
                        break;
                    }
                    result[localDataFeature[i].vectorOffset] = it->second;
                    break;
                }
            }
        }
    }
    return result; // 回傳轉換後的單筆資料特徵向量
}

// 回傳內部的特徵設定物件
DataFeature& Data::getDataFeature()
{
    return dataFeature;
}

// 取得整個資料集的二維矩陣
const vector<vector<double>>& Data::getData() const
{
    return data;
}

// 取得單筆資料的向量長度
unsigned int Data::getSingleDataSize() const
{
    return dataFeature.getSingleDataSize();
}

// 檢查單筆特徵向量是否符合預期的格式 (合法性檢查)
bool Data::format(const vector<double> &vec)
{
    const vector<VariableFeature> &localFeatrue = dataFeature.getFeatures();
    for(const VariableFeature &feature : localFeatrue)
    {
        switch(feature.type)
        {
            case CATEGORICAL:
            {
                switch(dataFeature.getEncodedType())
                {
                    case ONE_HOT_ENCODING:
                    {
                        // 檢查 One-hot 是否只有一個為 1
                        int count = 0;
                        for(int index = 0; index < feature.size; index++)
                        {
                            if(vec[feature.vectorOffset + index])
                                count++;
                            if(count > 1)
                            return false;
                        }
                        if(count < 0)
                            return false;
                        break;
                    }
                    case LABEL_ENCODING:
                    {
                        // 檢查是否為無效類別或越界
                        if(vec[feature.vectorOffset] == -1)//missing value
                            break;
                        
                        if(vec[feature.vectorOffset] >= feature.index.size() || vec[feature.vectorOffset] < 0)
                        {
                            cout<<"not CATEGORICAL\n";
                            cout<<vec[feature.vectorOffset]<<" : "<<feature.size<<endl;
                            return false;
                        }
                            
                        break;
                    }
                }
            }
            case NUMERICALLABEL:
            case NUMERICAL:
            {
                // 連續數值無限制 
                break;
            }
            case IGNORE:
            {
                //nothing
                break;
            }
            case BOOL:
            case BOOLLABEL:
            {
                // 檢查是否為正確的 0 或 1
                if(vec[feature.vectorOffset] != 0.0f && vec[feature.vectorOffset] != 1.0f)
                {
                    cout<<"not BOOL\n";
                    return false;
                }
                
                break;
            }
        }
    }
    return true;
}