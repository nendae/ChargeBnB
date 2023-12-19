#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
using namespace std;

double STATE_0_CUTOFF = 0.5;
double STATE_1_CUTOFF = 1.2;
double STATE_2_CUTOFF = 3;

int users;
int startTime; 
int endTime;

void getUsers() {
    cout << "How many users are in your Household? ";
    while (!(cin>>users)|| users < 1 ){
        cout << endl;
        cout << "Please input a valid number of users: ";
    }
}

/*  Asks the user for consumption values for each user in the household
    Modifies the State Values for each user based on thresholds defined above
*/
void getConsumption(int stateValues[][24])
{
    for(int i = 0; i < users; i++){
        cout << " ------- Input Consumption user " << i + 1 << "------" << endl;
        for(int j = 0; j < 24; j++){
            int h = 0;
            string time = "";

            if(j == 0){
                h = 12;
            } else if (j < 13){
                h = j;
            } else {
                h = j - 12;
            }

            if(j < 12){
                time = "AM";
            } else {
                time = "PM";
            }

            cout << "Input your energy consumption for " << h << time << ": ";
            double consumption;
            while (!(cin >> consumption) || (consumption < 0)){
                cout << endl;
                cout << "Please enter a number greater than 0";
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(),'\n');
            }


            if(consumption < STATE_0_CUTOFF){
                stateValues[i][j] = 0;
            } else if (consumption < STATE_1_CUTOFF){
                stateValues[i][j] = 1;
            } else if (consumption < STATE_2_CUTOFF){
                stateValues[i][j] = 2;
            } else {
                stateValues[i][j] = 3;
            }
        }
    }
}


/*  Fills out the transition matrix given the states of each user
    Spec Transitions is the transitions from one number to another
    Any Transitions is the transitions from one number to any other
    Transition matrix value is = spec/any at the given position
*/
void fillTransitionMatrix(int states[][24],double matrix[][4][4]){
    for(int i = 0; i < users; i++){
        int Spec_Transitions[4][4] = {0};
        int Any_Transitions[4] = {0};

        for(int j = 0; j < 23; j++){
            int current = states[i][j];
            int next = states[i][j+1];
            Spec_Transitions[current][next] += 1;
            Any_Transitions[current] += 1;
        }

        for(int m = 0; m < 4; m++){
            for(int n = 0; n < 4; n++){
                if (Any_Transitions[m] == 0){
                    matrix[i][m][n] =0;
                }
                else{
                    matrix[i][m][n] = (double)Spec_Transitions[m][n]/Any_Transitions[m];
                }
            }
        }
    }
}

/*  Fills out the probability matrix using each user's probability matrix and transition matrix
    Each row is based on the row before it
*/
void fillProbabilityMatrix(int states[][24], double probability[][24][4], double transition[][4][4]){
    
    for(int u = 0; u < users; u++){
        int initialState = states[u][0];
        probability[u][0][initialState] = 1; 

        for (int i = 1; i < 24; i++){
            for (int j = 0; j < 4; j++){
                for (int k = 0; k < 4; k++){
                    probability[u][i][j] += probability[u][i-1][k] * transition[u][k][j];
                }
            }
        }
    }
}

// Asks the user for a start and end time
void getHours(){
   cout << "At what time would it be okay to allow others to charge their vehicle? " << endl;
   while (!(cin >> startTime) || (startTime < 0 || startTime > 23)){
       cout << "Please input a whole number between 0 and 23" << endl;
       cin.clear();
       cin.ignore(numeric_limits<streamsize>::max(),'\n');
   }
   
   cout << "At what time would you like to end all charging sessions for vehicles? " << endl;
   while (!(cin >> endTime) || (endTime < 0 || endTime > 23)){
       cout << "Please input a whole number between 0 and 23" << endl;
       cin.clear();
       cin.ignore(numeric_limits<streamsize>::max(),'\n');
   }
}

/*  Algorithm for All Absent and All Asleep
    column 0 for absent, 1 for asleep
*/
double getAllAbsentOrAsleep (int column, int hour, double userMatrices[][24][4]){
    double result = 1; 
    for (int i = 0; i < users; i++){
        result *= userMatrices[i][hour][column];
    }

    return result; 
}

/*  Algorithm for All Active and All Hyper Active
    column 2 for active, 3 for hyper active
*/
double getAllActiveOrHyperActive(int column, int hour, double userMatrices[][24][4]){
    double result = 1; 
    for (int i = 0; i < users; i++){
        result *= (1 - userMatrices[i][hour][column]);
    }

    return 1 - result; 
}

//  Fills out the consumption probability matrix using the desired time frame
void getConsumptionProbability(int hours, double userMatrices[][24][4], double probability[][4]){
    for (int i = 0; i < hours; i++){
        for (int j = 0; j < 4; j++){
            for (int k = 0; k < users; k++){
                int time = (startTime+i) % 24;
                double result = 0; 
                if (j < 2){
                    result = getAllAbsentOrAsleep(j,time,userMatrices);
                }else{
                    result = getAllActiveOrHyperActive(j,time,userMatrices);
                }
                probability[i][j] = result; 
            }
        }
    }
}

// Returns the time with the highest probability of low consumption
int getBestHour(int hours, double probability[][4]){
    int bestHour = 0; 
    for (int i = 0; i < hours; i++){
        double bestProbability = probability[bestHour][0] + probability[bestHour][1];
        double currentProbability = probability[i][0] + probability[i][1]; 
        if (currentProbability > bestProbability){
            bestHour = i;
        }
    }
    return startTime + bestHour; 
}

int main() {
    
    getUsers();
    cout << "Users: " << users << endl;
    
    int UserStates[users][24] = {0};
    getConsumption(UserStates);

    cout << "------ State Matrix ------" << endl;
    for (int i = 0; i < users; i++){
        cout << "[ ";
        for(int j = 0; j < 24; j++){
            cout << fixed << setprecision(3) << UserStates[i][j] << " ";
        }
        cout << "]" << endl;
    }

    double UserTransitionMatrices[users][4][4] = {0};
    fillTransitionMatrix(UserStates, UserTransitionMatrices);

    cout << "------ Transition Matrices ------" << endl;
    for (int i = 0; i < users; i++){
        cout << "--- User " << i + 1 << " ---" << endl;
        for(int j = 0; j < 4; j++){
            cout << "[ ";
            for(int k = 0; k < 4; k++){
                cout << fixed << setprecision(3) << UserTransitionMatrices[i][j][k] << " ";
            }
            cout << "]" << endl;
        }
    }

    double UserProbabilityMatrices[users][24][4] = {0};
    fillProbabilityMatrix(UserStates, UserProbabilityMatrices, UserTransitionMatrices);

    cout << "------ Probability Matrices ------" << endl;
    for (int i = 0; i < users; i++){
        cout << "--- User " << i + 1 << " ---" << endl;
        for(int j = 0; j < 24; j++){
            cout << "[ ";
            for(int k = 0; k < 4; k++){
                cout << fixed << setprecision(3) << UserProbabilityMatrices[i][j][k] << " ";
            }
            cout << "]" << endl;
        }
    }

    getHours();
    int totalHours = (endTime - startTime + 24) % 24; 
    if(totalHours == 0){
        totalHours = 24;
    }

    double ConsumptionProbability[totalHours][4] = {0};
    getConsumptionProbability(totalHours,UserProbabilityMatrices,ConsumptionProbability);

    cout << "------ Consumption Matrix ------" << endl;
    for (int i = 0; i < totalHours; i++){
        cout << "[ ";
        for (int j = 0; j < 4; j++){
            cout << fixed << setprecision(3) << ConsumptionProbability[i][j] << " ";
        }
        cout << "]" << endl; 
    }
    cout << "-----------------";

    int BestHour = getBestHour(totalHours,ConsumptionProbability);
    
    string time = "";
    if (BestHour < 12){
        time = "AM";
    } else {
        time = "PM";
    }

    if (BestHour > 12){
        BestHour -= 12;
    } else if (BestHour == 0){
        BestHour = 12; 
    }
    cout << "Based on the information provided, " << BestHour << time << " is the best hour. ";
   
}
