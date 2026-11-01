# Atuadores do PRIMUS

Dados fornecidos pelo proprietário em 31/10/2026: bateria de 24 V, moteus-r4 da mjbots, motor equivalente ao usado no OpenQDD com enrolamento de 130 kV. A transmissão impressa foi modificada e é mecanicamente mais resistente; a folga é semelhante à do projeto de referência, mas não foi medida. A redução 9:1 é a já documentada neste repositório.

O [projeto original](https://www.aaedmusa.com/projects/openqdd) identifica um motor Eagle Power de 90 kV e controlador ODrive S1. Esses valores de desempenho não caracterizam o conjunto modificado do PRIMUS. A identificação exata do motor continua por confirmar; não inferimos um código de modelo apenas pela referência verbal.

## Hipóteses para a primeira simulação

Os dados e hipóteses ficam em [actuators.json](actuators.json). O Isaac e o gerador URDF leem esse mesmo perfil.

- Referência ideal de velocidade sem carga: 24 × 130 = 3120 rpm no rotor; com 9:1, cerca de 346,7 rpm ou 36,30 rad/s na junta. Não inclui queda de tensão, perdas ou enfraquecimento de campo. É usada como aproximação da curva torque–velocidade, não como medição.
- O limite provisório de 18 N.m na junta e a inércia refletida de 0,006 kg.m² foram preservados como hipóteses. A resistência mecânica maior não justifica aumentar o torque elétrico disponível.
- O limite de pico do controlador não equivale à corrente configurada ou à corrente contínua do motor. A [documentação do moteus](https://mjbots.github.io/moteus/troubleshooting/limited-torque/) descreve limites por corrente, tensão/back-EMF, potência, temperatura e comando de torque.
- A folga da transmissão ainda não é simulada: falta sua amplitude para escolher um modelo e uma faixa de randomização. A primeira tarefa valida equilíbrio ideal antes de acrescentar essas incertezas.
- A massa da bateria do CAD ainda corresponde à hipótese antiga de 0,68 kg; a informação de 24 V não permite recalcular sua massa. Nenhuma massa foi alterada automaticamente.

Quando houver acesso ao controlador real, um dump somente leitura da configuração/calibração pode fornecer corrente máxima, constante de torque calibrada, resistência, limites de velocidade e a relação rotor/saída configurada. Isso permite melhorar o modelo sem depender da ficha original do motor. Este trabalho não enviou comandos nem alterou firmware/configuração de nenhum moteus.
