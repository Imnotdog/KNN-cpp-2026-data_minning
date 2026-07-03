#include"Feature.h"

// VariableFeature 建構子：初始化特徵數值
VariableFeature::VariableFeature():name(""), type(IGNORE), vectorOffset(0), size(0), mean(0.0), variance(0.0)
{
    index.clear(); // 清空對應表
};

// 將傳入的 features 列表寫入 DataFeature 內部的 features
void DataFeature::setFeatures(const vector<VariableFeature>& copy)
{
    features = copy;
}

// 取得單筆資料展開成 vector 後的實際長度
unsigned int DataFeature::getSingleDataSize() const
{
    return singleDataSize;
}

// 從特徵列表中找出負責當作目標的特徵 (labelFeature)
VariableFeature DataFeature::getLable()
{
    // 如果已經緩存，直接回傳
    if(labelFeature.type == BOOLLABEL || labelFeature.type == NUMERICALLABEL)
    {
        return labelFeature;
    }  
    // 遍歷所有特徵，找出 BOOLLABEL 或 NUMERICALLABEL
    for(VariableFeature feature :features)
    {
        if(feature.type == BOOLLABEL || feature.type == NUMERICALLABEL)
        {
            labelFeature = feature;
            return labelFeature;
        }
    }
    cout<<"found no label\n"<<endl;
    exit(1);
}

// 取得所有特徵設定列表 (傳回參照以供外部更新平均與變異數)
vector<VariableFeature>& DataFeature::getFeatures()
{
    return features;
}

// 取得編碼方式
ENCODING DataFeature::getEncodedType() const
{
    return encodedType;
}

// 從給定的 attrib 陣列與資料表 Path 初始化特徵屬性對應表
DataFeature::DataFeature(const string Path, const vector<Attrib> &attrib, const ENCODING setEncodedType)
    :encodedType(setEncodedType),labelFeature()
{
    ifstream file(Path);
    if(file.fail())
    {
        cout<<"can't open "<<Path<<endl;
        exit(1);
    }

    // 準備讀取 attrib 檔案來建立類別變數的字典映射 (Map)
    char line[1000];
    string token;
    stringstream ss;
    unsigned int curIndex = 0;
    
    // 逐一檢查 attrib 並初始化 VariableFeature 結構
    for(unsigned int i=0; i<attrib.size() && !file.eof(); i++)
    {
        VariableFeature curFeature;
        curFeature.name = attrib[i].name;
        curFeature.type = attrib[i].type;
        curFeature.vectorOffset = curIndex; // 紀錄此特徵在 vector 裡的起始位置
        
        file.getline(line, 1000);
        ss.clear();
        ss.str(line);
        getline(ss, token, ':');
        
        // 跳過空白行
        if(token.empty()) {
            i--;        
            continue;   
        }
        
        // 防呆：確認檔案順序和 attrib 對應相同
        if(token != attrib[i].name)
        {
            cout<<"Name from attrib isn't smae as "<<Path<<" at line "<<i<<" : "<<attrib[i].name<<", "<<token<<endl;
            cout<<__FILE__<<" : "<<__LINE__<<endl;
            exit(1);
        }

        switch(curFeature.type)
        {
            case NUMERICALLABEL:
            case NUMERICAL:
            {
                // 數值型特徵只佔用一個維度
                curFeature.size = 1;
                break;
            }
            case IGNORE:
            {
                // 忽略的特徵佔用 0 個維度
                curFeature.size = 0;
                break;
            }
            case BOOL:
            case BOOLLABEL:
            {
                // 讀取冒號後面的各種可能類別（如 yes, no），幫他們編號
                while(getline(ss, token, ','))
                {
                    auto find = curFeature.index.find(token);
                    if(find == curFeature.index.end())
                    {
                        int temp = curFeature.index.size();
                        curFeature.index[token] = temp;
                    }
                }
                curFeature.size = 1;
                break;
            }
            case CATEGORICAL:
            {
                // 類別特徵處理
                while(getline(ss, token, ','))
                {
                    auto find = curFeature.index.find(token);
                    if(find == curFeature.index.end())
                    {
                        int temp = curFeature.index.size();
                        curFeature.index[token] = temp;
                    }
                }
                // 若使用 One-Hot Encoding，維度要加上類別數量
                if(encodedType == ONE_HOT_ENCODING)
                    curFeature.size = curFeature.index.size();
                else 
                    curFeature.size = 1;
                break;
            }
        }

        features.emplace_back(curFeature);
        curIndex +=curFeature.size;
    }
    singleDataSize = curIndex;
    file.close();

    // 為具有類別性質的特徵初始化統計陣列 (frequency, confidence)
    for(VariableFeature &feature : features)
    {
        switch(feature.type)
        {
            case BOOL: 
            case BOOLLABEL:
            case CATEGORICAL: 
            {
                feature.frequency = vector<unsigned int>(feature.index.size());
                feature.approveFrequency = feature.frequency;
                feature.confidence = vector<double>(feature.index.size());
                break;
            }
        }
    }
}

// 複製建構子
DataFeature::DataFeature(const DataFeature &copy):
encodedType(copy.encodedType), features(copy.features), singleDataSize(copy.singleDataSize), labelFeature()
{
    //nothing
}

// 比較運算子：確保兩個特徵架構一模一樣
bool DataFeature::operator ==(const DataFeature &comp) const
{
    if(encodedType != comp.encodedType || singleDataSize != comp.singleDataSize)
        return 0;
    int size = features.size();
    for(int i=0; i<size; i++)
    {
        if(features[i].name != comp.features[i].name)
            return 0;
    }
    return 1;
}

// 用於 debug，列印出所有特徵的定義
void DataFeature::showFeature() const
{
    int size = features.size();
    for(const VariableFeature &feature : features)
    {
        cout<<"Name : "<<feature.name<<endl;
        cout<<"Size : "<<feature.size<<endl;
        cout<<"Vector Offset : "<<feature.vectorOffset<<endl;
        cout<<"Type : "<<feature.type<<" = ";
        switch(feature.type)
        {
            case CATEGORICAL:
            {
                cout<<"Categorical"<<endl<<"Map size : "<<feature.index.size()<<endl;
                for(const auto& pair: feature.index)
                {
                    cout<<pair.first<<" : "<<pair.second<<endl;
                }
                break;
            }
            case BOOL:
            {
                cout<<"Bool"<<endl;
                for(const auto& pair: feature.index)
                {
                    cout<<pair.first<<" : "<<pair.second<<endl;
                }
                break;
            }
            case BOOLLABEL:
            {
                cout<<"Bool label"<<endl;
                for(const auto& pair: feature.index)
                {
                    cout<<pair.first<<" : "<<pair.second<<endl;
                }
                break;
            }
            case NUMERICALLABEL:
            {
                cout<<"Numerical label"<<endl;
                cout<<"mean : "<<feature.mean<<endl;
                cout<<"variance : "<<feature.variance<<endl;
                break;
            }
            case NUMERICAL:
            {
                cout<<"Numerical"<<endl;
                cout<<"mean : "<<feature.mean<<endl;
                cout<<"variance : "<<feature.variance<<endl;
                break;
            }
            case IGNORE:
            {
                cout<<"Ignored"<<endl;
                break;
            }
        }
        cout<<endl;
    }
    return ;
}