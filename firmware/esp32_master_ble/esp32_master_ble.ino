/*
 * sOFIa - ESP32 MAESTRA, variante BLE
 *
 * Mismo estandar de comunicacion que la variante WiFi/UDP: la trama no cambia
 * (sincronia, version, tipo, secuencia, longitud, carga, CRC-16). Solo cambia
 * el medio por el que viajan los bytes.
 *
 * Perfil: Nordic UART Service (NUS), el de facto para flujos de bytes sobre
 * BLE. Desde el punto de vista de la ESP32:
 *
 *     RX  6E400002-...  la PC ESCRIBE aqui, nosotros recibimos
 *     TX  6E400003-...  nosotros NOTIFICAMOS, la PC recibe
 *
 * Requiere la libreria NimBLE-Arduino (gestor de librerias de Arduino IDE).
 * Se usa NimBLE y no la pila BLE original de ESP32 porque ocupa alrededor de
 * un tercio de la memoria, y en esta placa la memoria se va a necesitar para
 * el control de los servos.
 *
 * QUE HAY QUE SABER DE BLE AQUI
 * -----------------------------
 * El MTU por defecto es 23 bytes, de los que solo 20 son utiles. Una trama
 * MATRIX16 son 24: no cabe. Por eso:
 *   - se pide un MTU de 247 al arrancar, y
 *   - la recepcion acumula en un buffer y extrae tramas completas, de modo
 *     que funcione igual si el sistema operativo del otro lado no concede
 *     el MTU grande y las tramas llegan partidas.
 * La PC hace exactamente lo mismo del otro lado.
 */

#include <NimBLEDevice.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

static const char* NOMBRE_BLE = "sOFIa-Maestra";

#define SERVICIO_UUID "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CARACT_RX     "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CARACT_TX     "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

static const uint8_t  SYNC0 = 0xA5, SYNC1 = 0x5A;
static const uint8_t  VERSION = 0x01;
static const uint8_t  MAX_PAYLOAD = 64;
static const uint32_t HEARTBEAT_TIMEOUT = 1500;

enum MsgType : uint8_t {
  MSG_HELLO = 0x01, MSG_ACK = 0x02, MSG_NACK = 0x03,
  MSG_PING  = 0x04, MSG_PONG = 0x05,
  MSG_GESTURE = 0x10, MSG_MATRIX16 = 0x11,
  MSG_HEARTBEAT = 0x20, MSG_STATUS = 0x21, MSG_STOP = 0x40
};

enum ErrCode : uint8_t {
  ERR_OK = 0x00, ERR_CRC = 0x01, ERR_TIPO = 0x02, ERR_LONGITUD = 0x03,
  ERR_BUS = 0x04, ERR_ESCLAVO = 0x05, ERR_VERSION = 0x06, ERR_PAYLOAD = 0x07
};

Adafruit_PWMServoDriver pwm1(0x40);
Adafruit_PWMServoDriver pwm2(0x41);

NimBLECharacteristic* carTX = nullptr;
bool      conectado = false;
uint8_t   target[16] = {0};
float     actual[16] = {0};
uint32_t  lastRx = 0;
uint8_t   txSeq = 0;

// Buffer de reensamblado: BLE puede partir una trama en varias
// notificaciones, o pegar dos tramas cortas en una sola.
static uint8_t  buf[256];
static size_t   buf_len = 0;

// ---------------------------------------------------------------- CRC
uint16_t crc16_ccitt(const uint8_t* d, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)d[i] << 8;
    for (uint8_t b = 0; b < 8; b++)
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

// ---------------------------------------------------------------- envio
void sendFrame(uint8_t type, const uint8_t* payload, uint8_t len) {
  if (carTX == nullptr || !conectado) return;
  uint8_t f[6 + MAX_PAYLOAD + 2];
  f[0] = SYNC0; f[1] = SYNC1;
  f[2] = VERSION; f[3] = type; f[4] = ++txSeq; f[5] = len;
  if (len) memcpy(f + 6, payload, len);
  uint16_t crc = crc16_ccitt(f + 2, 4 + len);
  f[6 + len] = crc & 0xFF;
  f[7 + len] = crc >> 8;
  carTX->setValue(f, 8 + len);
  carTX->notify();
}

void ack(uint8_t seq)  { sendFrame(MSG_ACK, &seq, 1); }
void nack(uint8_t seq, uint8_t err) {
  uint8_t p[2] = { seq, err };
  sendFrame(MSG_NACK, p, 2);
}

// ---------------------------------------------------------------- gestos
// 0x01 PUNO · 0x02 PAZ · 0x03 SALUDO · 0x04 PULSO · 0x05 GIRO
// `extra` es int8: la PC manda -1 como 0xFF.
void onGesture(uint8_t code, uint8_t conf, int8_t extra) {
  Serial.printf("GESTO %02X conf=%u extra=%d\n", code, conf, (int)extra);
  if (code == 0x05) {                 // GIRO: espejar el patron de la matriz
    for (int fila = 0; fila < 4; fila++)
      for (int col = 0; col < 2; col++) {
        int a = fila * 4 + col, b = fila * 4 + (3 - col);
        uint8_t t = target[a]; target[a] = target[b]; target[b] = t;
      }
  }
}

// ---------------------------------------------------------------- tramas
void procesarTrama(const uint8_t* f, uint8_t len_total) {
  uint8_t type = f[3], seq = f[4], len = f[5];
  const uint8_t* p = f + 6;

  lastRx = millis();

  if (f[2] != VERSION) { nack(seq, ERR_VERSION); return; }

  switch (type) {
    case MSG_HELLO: ack(seq); Serial.printf("HELLO seq=%u -> ACK\n", seq); break;

    // PING: se devuelve el payload EXACTO. El eco permite a la PC descartar
    // respuestas viejas y detectar si contesta otro dispositivo.
    case MSG_PING:
      sendFrame(MSG_PONG, p, len);
      Serial.printf("PING seq=%u -> PONG\n", seq);
      break;

    case MSG_STOP: memset(target, 0, 16); ack(seq); break;
    case MSG_HEARTBEAT: break;

    case MSG_MATRIX16:
      if (len == 16) memcpy(target, p, 16);
      else nack(seq, ERR_LONGITUD);
      break;                                   // flujo de estado: sin ACK

    case MSG_GESTURE:
      if (len >= 3) { onGesture(p[0], p[1], (int8_t)p[2]); ack(seq); }
      else nack(seq, ERR_LONGITUD);
      break;

    default:
      nack(seq, ERR_TIPO);
      break;
  }
}

/* Extrae todas las tramas completas que haya en el buffer. Aprovecha el doble
   byte de sincronia para descartar basura y recuperarse de un fragmento
   perdido, igual que FrameStream del lado de la PC. */
void drenarBuffer() {
  size_t i = 0;
  while (buf_len - i >= 8) {
    if (!(buf[i] == SYNC0 && buf[i + 1] == SYNC1)) { i++; continue; }

    uint8_t len = buf[i + 5];
    if (len > MAX_PAYLOAD) { i += 2; continue; }

    size_t total = 8 + len;
    if (buf_len - i < total) break;            // falta cola: esperar mas

    uint16_t rx = (uint16_t)buf[i + 6 + len] | ((uint16_t)buf[i + 7 + len] << 8);
    if (crc16_ccitt(buf + i + 2, 4 + len) != rx) { i += 2; continue; }

    procesarTrama(buf + i, total);
    i += total;
  }
  if (i) { memmove(buf, buf + i, buf_len - i); buf_len -= i; }
  if (buf_len > sizeof(buf) - 32) buf_len = 0;  // proteccion
}

// ---------------------------------------------------------------- BLE
class CallbacksServidor : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* s, NimBLEConnInfo& info) override {
    conectado = true;
    lastRx = millis();
    buf_len = 0;
    Serial.println("PC conectada");
  }
  void onDisconnect(NimBLEServer* s, NimBLEConnInfo& info, int reason) override {
    conectado = false;
    memset(target, 0, 16);                     // failsafe inmediato
    Serial.println("PC desconectada: modulos a reposo");
    NimBLEDevice::startAdvertising();
  }
};

class CallbacksRX : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& info) override {
    std::string v = c->getValue();
    if (buf_len + v.size() > sizeof(buf)) buf_len = 0;
    memcpy(buf + buf_len, v.data(), v.size());
    buf_len += v.size();
    drenarBuffer();
  }
};

// ---------------------------------------------------------------- servos
void applyServos() {
  for (int i = 0; i < 16; i++) {
    float d = (float)target[i] - actual[i];
    actual[i] += constrain(d, -12.0f, 12.0f);  // rampa: protege los servos
    uint16_t us = (uint16_t)(1000 + (actual[i] / 255.0f) * 1000.0f);
    (i < 8 ? pwm1 : pwm2).writeMicroseconds(i % 8, us);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pwm1.begin(); pwm1.setPWMFreq(50);
  pwm2.begin(); pwm2.setPWMFreq(50);

  NimBLEDevice::init(NOMBRE_BLE);
  NimBLEDevice::setMTU(247);                   // 24 bytes no caben en 23
  NimBLEDevice::setPower(ESP_PWR_LVL_P9);

  NimBLEServer* servidor = NimBLEDevice::createServer();
  servidor->setCallbacks(new CallbacksServidor());

  NimBLEService* servicio = servidor->createService(SERVICIO_UUID);

  carTX = servicio->createCharacteristic(
      CARACT_TX, NIMBLE_PROPERTY::NOTIFY);

  NimBLECharacteristic* carRX = servicio->createCharacteristic(
      CARACT_RX, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
  carRX->setCallbacks(new CallbacksRX());

  servicio->start();

  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICIO_UUID);          // para que la PC la reconozca
  adv->setName(NOMBRE_BLE);
  adv->start();

  Serial.printf("sOFIa Maestra BLE lista. Anunciando como '%s'\n", NOMBRE_BLE);
  Serial.println("Desde la PC:  python ble_ping.py");
}

void loop() {
  // Failsafe: sin trafico valido por 1.5 s, todo a reposo. Cubre que la PC
  // se cuelgue con el enlace BLE aun abierto.
  if (conectado && millis() - lastRx > HEARTBEAT_TIMEOUT) {
    if (memcmp(target, "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 16) != 0) {
      memset(target, 0, 16);
      Serial.println("failsafe: sin trafico, modulos a reposo");
    }
  }

  static uint32_t t = 0;
  if (millis() - t >= 20) { t = millis(); applyServos(); }   // 50 Hz
}
