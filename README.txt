# Projeto piloto para predição de demanda de produtos

Foram testados modelos de regressão e selecionado o com métricas melhores para predição de 1 a 3 meses
para quantidade de vendas (demanda)

Não foi possível utilizar o Databricks a priori, pois não possuo licensa (pessoal) e a versão gratuita mantém o Cluster 
ativo por apenas 2 horas, além de diversas limitações. Portanto, foi optado por utilizar o meu Spark instalado
em minha máquina pessoal. Algumas configurações de meu Windows não permitiram salvar tabelas com a função "saveAsTable"
do spark, então utilizei o Pandas para escrever os csv finais, obviamente não é escalável para volumes muito maiores.

Minhas conclusões finais sobre os modelos testados para as variáveis e dados fornecidos foram de que não foi possível obter
um modelo com métricas adequadas para seguir com a produtização, estudos posteriores poderiam explorar outras variáveis
no banco de dados da empresa, além de que, com o acesso a ferramentas de Auto ML, dentro do Databricks, a velocidade
dos estudos com certeza será superior a que eu tive em minha máquina pessoal, podendo investigar diversas correlações 
em bem menos tempo.

# Créditos
Caio Cezar V. Macedo