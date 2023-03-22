#include <iostream>
using namespace std;

class IOSys{
public:
    double process();
    char input;
    char output;

    IOSys(); //#1
    IOSys(double x); //#2

};

IOSys::IOSys(){ //#1
    cout<<"Input:None\nOutput:None"<<endl;
}

IOSys::IOSys(double x){ //#2
    input=x;
    output=process();
    cout<<"Input:"<<"Output:"<<endl;
}

double IOSys::process(){
    return ((input*input)/2);
}

int main(){
    IOSys ino; //#1

    IOSys inoo(6.0);//#2

}
