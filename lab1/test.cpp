#include <iostream>
#include <vector>
using namespace std;
  vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int top=0; int background=matrix.size()-1;
        int left=0;int right=matrix[0].size()-1;
        int i=0;int j=0;
        int direction=3; //0上 1下 2左 3右
    
        vector<int> re;
        while(left<=right&&top<=background){
           re.push_back(matrix[i][j]);
           if(i==top+1&&direction==0){
             direction=3;
             top++;
           }
           else if(i==background&&direction==1){
            background--;
            direction=2;
           }
           if(j==right&&direction==3){
            right--;
            direction=1;
           }
           if(j==left&&direction==2){
            direction=0;
            left++;
           }
           switch(direction){
              case 0: i--;break;
              case 1 : i++;break;
              case 2: j--;break;
              case 3 : j++;break;
           }
        }
        return re;
    }
    int main(){
    vector<vector<int>> matrix={{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    vector<int> re=spiralOrder(matrix);
    for(int i=0;i<re.size();i++){
        cout<<re[i]<<" ";
    }
    return 0;
    }
