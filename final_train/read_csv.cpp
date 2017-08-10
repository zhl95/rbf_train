// #include <string>
// #include <sstream>
// #include <iostream>
// #include <vector>
// #include <fstream>


// #include <string>
// #include <sstream>
// #include <iostream>
// #include <vector>
// #include <fstream>



#include "read_csv.h"





   vector< vector<string> > read_csv(string filename){
        ifstream in(filename);
        string line, field;
        vector<string> v;                // array of values for one line only
        vector< vector<string> > array;  // the 2D array


        while ( getline(in,line) )    // get next line in file
        {
            v.clear();
            stringstream ss(line);

            while (getline(ss,field,','))  // break line into comma delimitted fields
            {
                v.push_back(field);  // add each field to the 1D array
            }

            array.push_back(v);  // add the 1D array to the 2D array
        }

        return array;

        // print out what was read in

        // for (size_t i=0; i<array.size(); ++i)
        // {
        //     for (size_t j=0; j<array[i].size(); ++j)
        //     {
        //         cout << array[i][j] << "|"; // (separate fields by |)
        //     }
        //     cout << "\n";
        // }

    }


    
