#include <iostream>
using namespace std;

class IOSys{
    double input;
    double output;
public:
    IOSys(double x=0.5){ //#1 default : 0.5
    input=x;
    output=process(input);
    cout<< "Input:" << x << "\nOutput:" << process(x) <<endl;
    }; //#1 default 0.5
    ~IOSys() {}; // 소멸자
    double process(double X) {return ((X*X)/2);}
};

int main(){
    IOSys ino; //#1 default : 0.5
    IOSys inoo(6.0);// not default : 6.0
    
    IOSys *iosys=new IOSys(1);
    delete iosys;
    
    return 0;
}
