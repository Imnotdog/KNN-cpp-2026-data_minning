#include"shuffleData.h"

// 讀取檔案內容至 vector<string> 中，每一行作為一筆資料
void read(string path, vector<string> &data)
{
    ifstream file(path);
    if(file.fail())
    {
        cout<<"can't open "<<path<<endl;
        exit(1);
    }
    // 讀取直到檔案結束
    while(!file.eof())
    {
        string line;
        getline(file, line);
        data.push_back(line);
    }
    file.close();
}

// 將 vector<string> 中的資料一行行寫入指定路徑
void write(string path, vector<string> data)
{
    ofstream file(path);
    if(file.fail())
    {
        cout<<"can't open "<<path<<endl;
        exit(1);
    }
    // 逐行寫入，最後一行不換行
    for(int i=0; i<data.size(); i++)
    {
        file<<data[i];
        if(i != data.size()-1)
            file<<endl;
    }
    file.close();
}

// 讀取測試集與訓練集，將兩者合併洗牌後重新切割
void shullfeData(string TESTPATH, string TRAINPATH, int TRAINNUM)
{
    vector<string>  data;
    read(TESTPATH, data);  // 讀取測試資料
    read(TRAINPATH, data); // 接續讀取訓練資料

    // 設定隨機亂數種子並打亂資料順序
    random_device rd;
    mt19937 gen(rd());
    shuffle(data.begin(), data.end(), gen);

    // 將前面 TRAINNUM 筆寫回訓練集
    write(TRAINPATH, vector<string>(data.begin(), data.begin() + TRAINNUM));
    // 將剩下的寫回測試集
    write(TESTPATH, vector<string>(data.begin() + TRAINNUM + 1, data.end()));
    return ;
}