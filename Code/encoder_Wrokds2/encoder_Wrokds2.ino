#include <AccelStepper.h>
#include <Wire.h>

#define MPUaddr0 0x68    //addr used if pin AD0 is set to 0 (left unconnected)
#define TCAADDR 0x70

float AccelX, AccelY, AccelZ = 0;
float accelXangle, accelYangle = 0;
float pitch, roll, pitch_now, roll_now = 0;
float beta = 0.98;

float xa, ya, za;

const int numSteppers = 3;

int magnetStatus = 0;
int lowbyte;
word highbyte;
int rawAngle;

double degAngles[numSteppers];
int quadrantNumbers[numSteppers];
int previousQuadrantNumbers[numSteppers];
double numberOfTurns[numSteppers];
double correctedAngles[numSteppers];
double startAngles[numSteppers];
double totalAngles[numSteppers];
double previousTotalAngles[numSteppers];
float finalAngles[numSteppers];

AccelStepper steppers[numSteppers] = {
  AccelStepper(AccelStepper::DRIVER, 9, 8), // base stepper
  AccelStepper(AccelStepper::DRIVER, 13, 12),   // left stepper
  AccelStepper(AccelStepper::DRIVER, 5, 4),   // right stepper
};

const int stepsPerRevolution[numSteppers] = {800, 1600, 1600};
float degreesToSteps[numSteppers];
const int maxSpeeds[numSteppers] = {3950, 3950, 3950};
const int accelerations[numSteppers] = {3950, 3950, 3950};
int maxSteps[numSteppers];
unsigned long previousTime = 0;
const long interval = 50;

double kp[numSteppers] = {3, 0.3, 0.3};
double ki[numSteppers] = {0, 0, 0};
double kd[numSteppers] = {0, 0, 0};

double setpoints[numSteppers + 1] = {0, 0, 0, 1};
double input[numSteppers];
double output[numSteppers];

int sensorPin[3] = {3, 7, 6};

double integral[numSteppers] = {0, 0, 0};
double previousError[numSteppers] = {0, 0, 0};

int reset = 0;
bool caliberation_done = false;
unsigned long startTime;

void tcaselect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(0x70);
  Wire.write(1 << i);
  Wire.endTransmission();
}

void writeToStream(){
  for (int i = 0; i < numSteppers; i++) {
    if(i == 0) Serial.print(pitch);
    else Serial.print(finalAngles[i]);
    if (i < numSteppers - 1) {
      Serial.print(",");
    }
  }
  Serial.println();
}

void newWriteToStream(double time){
  for (int i = 0; i < numSteppers + 1; i++) {
    Serial.print(setpoints[i]);
    if (i < numSteppers + 1) {
      Serial.print(",");
    }
  }
  Serial.print(time);
  Serial.println();
}

void readFromStream(){
  String response = Serial.readStringUntil('\n');
  int index = 0;
  char *token = strtok((char *)response.c_str(), ",");
  while (token != NULL) {
    setpoints[index++] = atoi(token);
    token = strtok(NULL, ",");
  }
}

void updateSensorData(int sensor, int setup=0) {
  if(sensor == 0){
    // tcaselect(3);
    if(reset == 0){
      Wire.beginTransmission(MPUaddr0);
      Wire.write(0x6B); // PWR_MGMT_1 register 
      Wire.write(0); // set to zero (wakes up the MPU-6050)
      Wire.endTransmission(true);
      reset = 1;
    }

    readMPU(); 
  }
  else{
    if(setup == 0){
      ReadRawAngle(sensor);
      correctAngle(sensor);
      checkQuadrant(sensor);
      finalAngles[sensor] = totalAngles[sensor];
    }
    else{
      checkMagnetPresence(sensor);
      ReadRawAngle(sensor);
      startAngles[sensor] = degAngles[sensor];
      correctAngle(sensor);
      finalAngles[sensor] = correctedAngles[sensor];
      checkQuadrant(sensor);
    }
  }
}


void setup() {
  Wire.begin();
  Serial.begin(115200);

  Wire.setClock(800000);
  Wire.setWireTimeout(3000, true);
  // Serial.println("setup start");

  Wire.beginTransmission(MPUaddr0);
  Wire.write(0x6B); // PWR_MGMT_1 register 
  Wire.write(0); // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);

  for (int i = 0; i < numSteppers; i++) {
    degreesToSteps[i] = stepsPerRevolution[i] / 360.0;
    maxSteps[i] = degreesToSteps[i] * 180;
    steppers[i].setMaxSpeed(1000);
    steppers[i].setAcceleration(500);
  }
  Serial.println("start");
  // for (int i = 0; i < numSteppers; i++) {
  //   updateSensorData(i, 1);
  // }
    // updateSensorData(0, 1);
    // updateSensorData(1, 1);
    // updateSensorData(2, 1);

  Serial.println("end");

  startTime = millis();
  delay(100);
  // Serial.println("setup end");

  // // Handshake: Signal to PC that Arduino is ready
  // Serial.println("READY");

  // // Wait for the PC to send the start signal
  // while (Serial.available() == 0) {}
  // String startSignal = Serial.readStringUntil('\n');
  // if (startSignal == "START") {
  //   // Communication started
  // }
}

void computePID(int i, float delta, int cond = 0){
  input[i] = finalAngles[i];
  double error;

  if(cond == 0) error = setpoints[i];
  else {
    error = setpoints[i] - input[i];
  }

  integral[i] += error * (delta / 1000.0);
  double derivative = (error - previousError[i]) / (delta / 1000.0);
  output[i] = kp[i] * error + ki[i] * integral[i] + kd[i] * derivative;
  previousError[i] = error;

  output[i] = constrain(output[i], -maxSpeeds[i], maxSpeeds[i]);
  // if(abs(setpoints[i]) < 2){
  //   steppers[i].setSpeed(0);
  //   steppers[i].runSpeed();
  //   return;

  // }
  steppers[i].setSpeed(output[i]);
  steppers[i].runSpeed();
}

void loop() {
  unsigned long currentTime = millis();
  double elapsedTime = (currentTime - startTime) / 1000.0;
  double delta = 0.1;

  if (Serial.available() > 0) {
    readFromStream();
  }  

  int cal_speed[3] = {0, 50, 50};

  float start[3] = {0, 0, 0};
  if(int(setpoints[3]) == 4){
    setpoints[3] = 2;
    if(!caliberation_done) {
      caliberation_done = true;
      Serial.println("6969");  
      return;
    };
    for(int i = 0; i < numSteppers; i++){
      setpoints[i] = 0;
      degAngles[i] = start[i];
      correctedAngles[i] = start[i];
      numberOfTurns[i] = 0;
      totalAngles[i] = (numberOfTurns[i] * 360) + correctedAngles[i];
      finalAngles[i] = totalAngles[i];
    }

  steppers[1].setCurrentPosition(0);
  steppers[2].setCurrentPosition(0); 

  float desiredAngles[numSteppers] = {0.0, 90.0, -90.0};

    for (int i = 1; i < numSteppers; i++) { 
      int steps = (desiredAngles[i] / 360.0) * stepsPerRevolution[i];
      // steppers[i].moveTo(steps); 
      steppers[i].runToNewPosition(steps);
    }
    Serial.println("6969");
    delay(1000);
    return;
  }

  for (int i = 0; i < numSteppers; i++) {
    updateSensorData(i);

    if(int(setpoints[3]) == 1){
      if(i == 0){
        steppers[0].setSpeed(0);
        steppers[0].runSpeed();
        continue;
      };
      if(int(setpoints[i]) == 6969){
        steppers[i].setSpeed(cal_speed[i]);
        steppers[i].runSpeed();
        continue;
      }
    }
    computePID(i, delta);
  }

  currentTime = millis();
  if (currentTime - previousTime >= interval) {
    previousTime = currentTime;
    writeToStream();
    // newWriteToStream(currentTime);
  }
  delayMicroseconds(delta * 1000);
}


void readMPU(){
  
  Wire.beginTransmission(MPUaddr0);
  Wire.write(0x3B); //send starting register address
  Wire.endTransmission(false); //restart for read
  Wire.requestFrom(MPUaddr0, 6, true); //get six bytes accelerometer data

  xa = Wire.read() << 8 | Wire.read();
  ya = Wire.read() << 8 | Wire.read();
  za = Wire.read() << 8 | Wire.read();
  // formula from https://wiki.dfrobot.com/How_to_Use_a_Three-Axis_Accelerometer_for_Tilt_Sensing
  pitch_now = (atan(-1 * xa / sqrt(pow(ya, 2) + pow(za, 2))) * 180 / PI); 
  roll_now = (atan(ya / sqrt(pow(xa, 2) + pow(za, 2))) * 180 / PI) ;

  pitch = beta * pitch + (1 - beta) * pitch_now;
  roll = beta * roll + (1 - beta) * roll_now;

}


void setAccelSensitivity(uint8_t g){
  //Config AFS_SEL[1:0] bits 4 and 3 in register 0x1C
  //0x00: +/-2g (default)
  //0x08: +/-4g
  //0x10: +/-8g
  //0x18: +/-16g

  Wire.beginTransmission(MPUaddr0);   //initialize comm with MPU @ 0x68
  Wire.write(0x1C);                   //write to register 0x1C
  Wire.write(g);                      //setting bit 7 to 1 resets all internal registers to default values
  Wire.endTransmission();             //end comm
}

void resetMPU(void){
  Wire.beginTransmission(MPUaddr0);   //initialize comm with MPU @ 0x68
  Wire.write(0x6B);                   //write to register 0x6B
  Wire.write(0x00);                   //reset all internal registers to default values
  Wire.endTransmission();             //end comm
  delay(100);
}

void readAccel(float accelDivisor){
  //NOTE: as you increase the accelerometer's range, the resolution decreases
  //+/-2g, use divisor of 16384 (14-bit resolution)
  //+/-4g, use divisor of 8192 (13-bit resolution)
  //+/-8g, use divisor of 4096 (12-bit resolution)
  //+/-16g, use divisor of 2048 (11-bit resolution)
  
  Wire.beginTransmission(MPUaddr0);
  Wire.write(0x3B);
  Wire.endTransmission();
  Wire.requestFrom(MPUaddr0, 6);    //read 6 consecutive registers starting at 0x3B
  if (Wire.available() >= 6){
    int16_t temp0 = Wire.read() << 8;   //read upper byte of X
    int16_t temp1 = Wire.read();        //read lower byte of X
    AccelX = (float) (temp0 | temp1);
    AccelX = AccelX / accelDivisor;
    
    temp0 = Wire.read() << 8;           //read upper byte of Y
    temp1 = Wire.read();                //read lower byte of Y
    AccelY = (float) (temp0 | temp1);
    AccelY = AccelY / accelDivisor;
    
    temp0 = Wire.read() << 8;           //read upper byte of Z
    temp1 = Wire.read();                //read lower byte of Z
    AccelZ = (float) (temp0 | temp1);
    AccelZ = AccelZ / accelDivisor;
        
  }
  //You can only calculate roll and pitch from accelerometer data
  accelXangle = (atan2(AccelY, AccelZ)) * 180 / PI;                                   //calculate roll
  accelYangle = (atan2(-AccelX, sqrt(pow(AccelY, 2) + pow(AccelZ, 2)))) * 180 / PI;   //calculate pitch
}


void ReadRawAngle(int sensor) {
  tcaselect(sensorPin[sensor]);

  Wire.beginTransmission(0x36);
  Wire.write(0x0D);
  Wire.endTransmission();
  Wire.requestFrom(0x36, 1);
  while (Wire.available() == 0);
  lowbyte = Wire.read();

  Wire.beginTransmission(0x36);
  Wire.write(0x0C);
  Wire.endTransmission();
  Wire.requestFrom(0x36, 1);
  while (Wire.available() == 0);
  highbyte = Wire.read();

  highbyte = highbyte << 8;
  rawAngle = highbyte | lowbyte;
  degAngles[sensor] = rawAngle * 0.087890625;
}

void correctAngle(int sensor) {
  correctedAngles[sensor] = 360.0 - degAngles[sensor];
}

void checkQuadrant(int sensor) {
  if (correctedAngles[sensor] >= 0 && correctedAngles[sensor] <= 90) quadrantNumbers[sensor] = 1;
  else if (correctedAngles[sensor] > 90 && correctedAngles[sensor] <= 180) quadrantNumbers[sensor] = 2;
  else if (correctedAngles[sensor] > 180 && correctedAngles[sensor] <= 270) quadrantNumbers[sensor] = 3;
  else if (correctedAngles[sensor] > 270 && correctedAngles[sensor] < 360) quadrantNumbers[sensor] = 4;

  if (quadrantNumbers[sensor] != previousQuadrantNumbers[sensor]) {
    if (quadrantNumbers[sensor] == 1 && previousQuadrantNumbers[sensor] == 4) numberOfTurns[sensor]++;
    if (quadrantNumbers[sensor] == 4 && previousQuadrantNumbers[sensor] == 1) numberOfTurns[sensor]--;
    previousQuadrantNumbers[sensor] = quadrantNumbers[sensor];
  }

  totalAngles[sensor] = (numberOfTurns[sensor] * 360) + correctedAngles[sensor];
}

void checkMagnetPresence(int sensor) {
  tcaselect(sensorPin[sensor]);
  while ((magnetStatus & 32) != 32) {
    magnetStatus = 0;
    Wire.beginTransmission(0x36);
    Wire.write(0x0B);
    Wire.endTransmission();
    Wire.requestFrom(0x36, 1);
    while (Wire.available() == 0);
    magnetStatus = Wire.read();
  }
}



