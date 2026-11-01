# Primeiro PPO de terreno no Isaac (2026-10-31) — post-mortem

Execucao `2026-10-31_15-16-02-terrain-cuda-0`: tarefa `Primus-Terrain-Direct`, 128 ambientes, 3.000 atualizacoes,
acao = torque bruto (+-18 N.m) nos 6 motores. Os artefatos foram removidos na limpeza do workspace; o resultado
foi considerado ruim e substituido pelo RL residual em `sim/warp_rl/`.

**O que aconteceu:** a politica nao aprendeu a equilibrar. A trajetoria gravada mostra o trem de pouso descendo
(-97/+97 graus -> 0) e o robo sentado nele, inclinado 12-56 graus, com a base a ~0,28 m (nominal 0,38 m).

**Causas:**
- o alvo de posicao do servo do trem nunca foi definido (o Isaac Lab inicia `joint_pos_target` com zeros = abaixado);
- terminacao frouxa (altura < 0,20 m, inclinacao > 60 graus) e recompensa `cos(inclinacao)` + bonus de vida grande:
  sentado e torto ainda rendia ~80% da recompensa;
- curriculum por ponto de nascimento deixou quase todos os ambientes depois da escada;
- torque bruto com desvio de exploracao crescente (0,30 -> 0,52), sem compensacao de gravidade;
- actor sem os estados que o LQR usa (dx, theta, L0); atrito padrao do PhysX (0,5) no terreno; roda como cilindro
  sem geometria exata.

Essas licoes valem para a volta ao Isaac (degraus grandes + camera). A direcao atual esta em
[direcao_controle.md](direcao_controle.md).
