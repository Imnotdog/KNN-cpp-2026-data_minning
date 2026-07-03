#pragma once
#include<iostream>
#include<fstream>
#include<unordered_map>
#include<vector>
#include<string>
#include<sstream>

using namespace std;

// 支援的特徵型態列舉
enum DataType
{
    NUMERICAL,        // 連續數值
    CATEGORICAL,      // 類別 (目前支援 one-hot / label encoding)
    BOOL,             // 布林值
    IGNORE,           // 忽略不使用的字串/特徵
    BOOLLABEL,        // 雙類別目標標籤 (存活/死亡 等)
    NUMERICALLABEL    // 數值型目標標籤 (回歸用)
    //MULTICLASSLABEL
    //TODO: add MULTILABEL
};

// 類別特徵的編碼方式
enum ENCODING
{
    ONE_HOT_ENCODING, // 獨熱編碼 (將類別展開成多個特徵)
    LABEL_ENCODING    // 標籤編碼 (將類別轉為數字)
    /*,
    FREQUENCY_ENCODING*/
};

// 定義每個特徵的名稱與型態
struct Attrib
{
    string name;      // 特徵名稱
    DataType type;    // 特徵型態
};

// 儲存每個特徵在轉換後的統計資訊與格式
struct VariableFeature
{
    string name;               // 特徵名稱
    DataType type;             // 型態
    unsigned int vectorOffset; // 在資料 vector 中的起始索引
    unsigned int size;         // 特徵佔用的長度 (One-hot 會有長度)

    // 針對連續數值 (NUMERICAL) 的統計資訊 (用作標準化)
    double mean;      // 平均值
    double variance;  // 變異數
    
    // 針對類別 (CATEGORICAL) 或 布林 (BOOL) 的資訊
    unordered_map<string, int> index;  // 紀錄字串與整數索引的對應 (編碼用)
    vector<double> confidence;         // 該類別的目標信心水準 (目標存活率)
    vector<unsigned int> frequency;    // 該類別出現的總次數
    vector<unsigned int> approveFrequency; // 該類別中對應為正向標籤 (1.0) 的次數

    VariableFeature();
};

// 統整並管理資料集所有特徵
class DataFeature
{
private:
    const ENCODING encodedType;        // 使用的編碼方法
    vector<VariableFeature> features;  // 所有特徵的列表
    VariableFeature labelFeature;      // 目標標籤特徵 (Label)
    unsigned int singleDataSize;       // 單筆資料轉換為 vector 後的總長度
public:
    DataFeature(const string typePath, const vector<Attrib> &attrib, const ENCODING getEncodeType = LABEL_ENCODING);
    DataFeature(const DataFeature &copy);
    
    void setFeatures(const vector<VariableFeature>& copy);
    unsigned int getSingleDataSize() const;
    VariableFeature getLable();
    vector<VariableFeature> &getFeatures(); // 回傳特徵列表的參照 (允許修改)
    ENCODING getEncodedType() const;
    bool operator ==(const DataFeature &comp) const;
    void showFeature() const;
};