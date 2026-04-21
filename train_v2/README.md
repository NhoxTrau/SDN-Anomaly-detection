# train_v2 (InSDN / OpenFlow-safe refactor v3)

## Muc tieu
- dung InSDN lam dataset train chinh
- giu runtime theo huong `sdn_realtime`
- bo hoan toan delta gia tao tu InSDN CSV
- tach ro:
  - **ML-core features** de train model
  - **runtime rules** de xu ly burst, scan, false positive

## Feature scheme mac dinh
`insdn_ml_core_v1`

### ML-core features (train + runtime)
- `packet_count`
- `byte_count`
- `flow_duration_s`
- `packet_rate`
- `byte_rate`
- `avg_packet_size`
- `src_port_norm`
- `dst_port_norm`
- `protocol_tcp`
- `protocol_udp`
- `protocol_icmp`

### Runtime-only rule features
- `packet_delta`
- `byte_delta`
- `packet_rate_delta`
- `byte_rate_delta`

## Sequence policy
- `seq_len = 4`
- sequence key theo service-centric logic:
  - offline: `(src_ip, dst_ip, dst_port, protocol)`
  - runtime: them `dpid` vao key de tranh trung giua switch

## 3 model trong de tai
- `lstm`
- `transformer`
- `lstm_autoencoder_insdn`

## Ghi chu
- `insdn_openflow_v1`, `telemetry_v2`, `telemetry_v1` duoc giu nhu alias tuong thich nguoc, nhung deu tro ve cung 11 ML-core features
- bundle va scaler cu theo CICIDS khong con phu hop
