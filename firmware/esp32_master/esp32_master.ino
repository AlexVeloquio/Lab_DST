/*
 * sOFIa - ESP32 MAESTRA
 * Receptor UDP del protocolo del Eje SC -> PCA9685 -> servos.
 *
 * Contraparte de core/protocol.py. Implementa:
 *   - parser con resincronizacion por doble byte de sync
 *   - verificacion CRC-16/CCITT-FALSE
 *   - ACK de los mensajes confiables (HELLO, GESTURE, STOP)
 *   - failsafe: si no llega heartbeat en HEARTBEAT_TIMEOUT, todo a reposo
 *
 * Requiere la libreria Adafruit_PWMServoDriver.
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

static const char* SSID = "SOFIA_AP";
static const char* PASS = "cambiar_esto";
static const uint16_t PORT = 4210;

static const uint8_t  SYNC0 = 0xA5, SYNC1 = 0x5A;
static const uint8_t  VERSION = 0x01;
static const uint8_t  MAX_PAYLOAD = 64;
static const uint32_t HEARTBEAT_TIMEOUT = 1500;   // ms sin PC -> reposo

enum MsgType : uint8_t {
  MSG_HELLO = 0x01, MSG_ACK = 0x02, MSG_NACK = 0x03,
  MSG_GESTURE = 0x10, MSG_MATRIX16 = 0x11,
  MSG_HEARTBEAT = 0x20, MSG_STATUS = 0x21, MSG_STOP = 0x40
};

WiFiUDP udp;
Adafruit_PWMServoDriver pwm1(0x40);
Adafruit_PWMServoDriver pwm2(0x41);

uint8_t  rxBuf[128];
uint8_t  target[16] = {0};      // comando recibido, 0-255 por modulo
float    actual[16] = {0};      // valor aplicado, con rampa
uint32_t lastRx = 0;
uint8_t  txSeq = 0;

uint16_t crc16_ccitt(const uint8_t* d, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)d[i] << 8;
    for (uint8_t b = 0; b < 8; b++)
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

void sendFrame(uint8_t type, const uint8_t* payload, uint8_t len) {
  uint8_t f[6 + MAX_PAYLOAD + 2];
  f[0] = SYNC0; f[1] = SYNC1;
  f[2] = VERSION; f[3] = type; f[4] = ++txSeq; f[5] = len;
  memcpy(f + 6, payload, len);
  uint16_t crc = crc16_ccitt(f + 2, 4 + len);
  f[6 + len] = crc & 0xFF; f[7 + len] = crc >> 8;
  udp.beginPacket(udp.remoteIP(), udp.remotePort());
  udp.write(f, 8 + len);
  udp.endPacket();
}

void ack(uint8_t seq) { sendFrame(MSG_ACK, &seq, 1); }

// Codigos de gesto. `extra` es un entero CON SIGNO (int8_t): la PC envia -1
// como 0xFF, asi que hay que leerlo como int8_t y no como uint8_t.
//   0x01 PUNO     (estatica)
//   0x02 PAZ      (estatica)
//   0x03 SALUDO   (dinamica)
//   0x04 PULSO    (dinamica)
//   0x05 GIRO     (dinamica)  extra: -1 quedo en DORSO, +1 quedo en PALMA
void onGesture(uint8_t code, uint8_t conf, int8_t extra) {
  Serial.printf("GESTO %02X conf=%u extra=%d\n", code, conf, (int)extra);

  switch (code) {
    case 0x05:                         // GIRO: espejar el patron de la matriz
      // Solo las dos primeras columnas: recorrer las cuatro intercambiaria
      // cada par dos veces y la matriz quedaria igual que antes.
      for (int fila = 0; fila < 4; fila++) {
        for (int col = 0; col < 2; col++) {
          int a = fila * 4 + col, b = fila * 4 + (3 - col);
          uint8_t t = target[a]; target[a] = target[b]; target[b] = t;
        }
      }
      break;
    default:
      break;                         // los demas patrones van aqui
  }
}

void applyServos() {
  // Rampa suave hacia el objetivo: segunda linea de defensa contra el
  // chattering, por si la PC llegara a enviar comandos ruidosos.
  for (int i = 0; i < 16; i++) {
    float d = (float)target[i] - actual[i];
    actual[i] += constrain(d, -12.0f, 12.0f);
    uint16_t us = (uint16_t)(1000 + (actual[i] / 255.0f) * 1000.0f);  // 1000-2000 us
    (i < 8 ? pwm1 : pwm2).writeMicroseconds(i % 8, us);
  }
}

void handleFrame(const uint8_t* f, int n) {
  if (n < 8 || f[0] != SYNC0 || f[1] != SYNC1) return;
  uint8_t type = f[3], seq = f[4], len = f[5];
  if (len > MAX_PAYLOAD || n < 8 + len) return;

  uint16_t rx = (uint16_t)f[6 + len] | ((uint16_t)f[7 + len] << 8);
  if (crc16_ccitt(f + 2, 4 + len) != rx) return;     // trama corrupta: se ignora

  lastRx = millis();
  const uint8_t* p = f + 6;

  switch (type) {
    case MSG_HELLO:     ack(seq); break;
    case MSG_STOP:      memset(target, 0, 16); ack(seq); break;
    case MSG_HEARTBEAT: break;
    case MSG_MATRIX16:  if (len == 16) memcpy(target, p, 16); break;
    case MSG_GESTURE:   if (len >= 3) { onGesture(p[0], p[1], (int8_t)p[2]); ack(seq); } break;
    default: break;
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pwm1.begin(); pwm1.setPWMFreq(50);
  pwm2.begin(); pwm2.setPWMFreq(50);

  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.println();
  Serial.print("IP de la maestra: "); Serial.println(WiFi.localIP());
  udp.begin(PORT);
}

void loop() {
  int n = udp.parsePacket();
  if (n > 0) {
    int len = udp.read(rxBuf, sizeof(rxBuf));
    handleFrame(rxBuf, len);
  }

  if (millis() - lastRx > HEARTBEAT_TIMEOUT) memset(target, 0, 16);  // failsafe

  static uint32_t t = 0;
  if (millis() - t >= 20) { t = millis(); applyServos(); }           // 50 Hz
}
