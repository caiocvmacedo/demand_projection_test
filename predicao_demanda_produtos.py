# Código para predição de demanda
# Versão 0.0.1

# Parâmetros:

# Tabelas
tb_produtos = 'teste_petz/Produtos.csv'
tb_vendas = 'teste_petz/Vendas.csv'
tb_lojas = 'teste_petz/Lojas.csv'
tb_unidades = 'teste_petz/Unidades_Negocios.csv'
tb_canais = 'teste_petz/Canais.csv'
# Separador dos arquivos csv
separador = ';'
# Quantidade de meses a serem analisados na modelagem
num_meses = 6
# Local de armazenamento das predições
predict_30_save_local = 'teste_petz/saves/predicts/predicao_demanda_30_dias.csv'
predict_60_save_local = 'teste_petz/saves/predicts/predicao_demanda_60_dias.csv'
predict_90_save_local = 'teste_petz/saves/predicts/predicao_demanda_90_dias.csv'

# imports
import findspark
import os
from pyspark.sql.functions import *
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.stat import Correlation
from pyspark.sql import Window

# Sessão spark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master('local[*]') \
    .appName("Iniciando com Spark") \
    .getOrCreate()

# Carregando informações
produtos = spark.read.csv(tb_produtos, sep = separador, header = True)
vendas = spark.read.csv(tb_vendas, sep = separador, header = True)
lojas = spark.read.csv(tb_lojas, sep = separador, header = True)
unidades = spark.read.csv(tb_unidades, sep = separador, header = True)
canais = spark.read.csv(tb_canais, sep = separador, header = True)

# Preparação inicial
dataset = vendas.withColumn('qtde_venda', regexp_replace('qtde_venda', ',', '.').cast('float'))\
        .withColumn('valor_venda', regexp_replace('valor_venda', ',', '.').cast('float'))\
        .withColumn('valor_imposto', regexp_replace('valor_imposto', ',', '.').cast('float'))\
        .withColumn('valor_custo', regexp_replace('valor_custo', ',', '.').cast('float'))\
        .withColumn('data', concat_ws('-', substring(col('id_data').cast('date'), 1, 4), 
                                      substring(col('id_data').cast('date'), 6, 2)))\
        .join(produtos.withColumnRenamed('produto', 'id_produto'), on='id_produto', how='left')\
        .withColumn('flg_prodt', when(isnull(col('produto_nome')), 0).otherwise(1))\
        .join(lojas, on='id_loja', how='left').withColumn('flg_loja', when(isnull(col('cod_loja')), 0).otherwise(1))\
        .join(unidades, on='id_unidade_negocio', how='left').withColumn('flg_unidade', when(isnull(col('unidade_negocio')), 
                                                                                           0).otherwise(1))\
        .join(canais.withColumnRenamed('cod_canal', 'id_canal'), on='id_canal', how='left')\
        .withColumn('flg_canal', when(isnull(col('canal')), 0).otherwise(1))

# Enumerando clientes
cliente = dataset.select('id_cliente').distinct().withColumn('cliente', concat_ws('_', lit('cliente'), 
                                                                                  monotonically_increasing_id()))

# Enumerando tipos de clientes
tipo_cliente = dataset.select('id_tipo_cliente').distinct().withColumn('tipo_cliente', 
                                                                    concat_ws('_', lit('tipo_cliente'), 
                                                                              monotonically_increasing_id()))

# Sintetizando dados em uma nova versão
dataset_v02 = dataset.join(cliente, on='id_cliente', how='left').join(tipo_cliente, on='id_tipo_cliente', how='left')\
                    .drop('id_produto', 'id_loja', 'id_unidade_negocio', 'id_canal', 'id_cupom', 
                          'id_cliente', 'id_endereco_venda', 'id_tipo_cliente', 'id_data').distinct().cache()

# Correção de variáveis e nulos

dataset_v03 = dataset_v02.where(col('qtde_venda') >= 0).where(col('valor_venda') > 0)\
    .where(col('valor_imposto') > 0).where(col('valor_custo') > 0)\
    .withColumn('fornecedor', when(substring(col('fornecedor'), 1, 4) != 'Forn', None).otherwise(col('fornecedor')))\
    .withColumn('produto_nome', when(substring(col('produto_nome'), 1, 4) != 'Prod', None).otherwise(col('produto_nome')))\
    .withColumn('categoria', when(substring(col('categoria'), 1, 4) != 'Cate', None).otherwise(col('categoria')))\
    .withColumn('sub_categoria', when(substring(col('sub_categoria'), 1, 3) != 'Sub', None).otherwise(col('sub_categoria')))\
    .withColumn('cod_loja', when(substring(col('cod_loja'), 1, 4) != 'Loja', None).otherwise(col('cod_loja')))\
    .withColumn('regional', when(substring(col('regional'), 1, 4) != 'Regi', None).otherwise(col('regional')))\
    .withColumn('distrito', when(substring(col('distrito'), 1, 4) != 'Dist', None).otherwise(col('distrito')))\
    .withColumn('cliente', when(substring(col('cliente'), 1, 4) != 'clie', None).otherwise(col('cliente')))\
    .withColumn('tipo_cliente', when(substring(col('tipo_cliente'), 1, 4) != 'tipo', None).otherwise(col('tipo_cliente')))\
    .cache()

# Dropando dados inconsistentes
dataset_v04 = dataset_v03.dropna()

# Agregando variáveis numéricas
dataset_numeric_agg = dataset_v04.select('produto_nome', 'data', 
                                'qtde_venda', 'valor_venda', 
                                'valor_imposto', 'valor_custo').groupBy('produto_nome', 'data').sum()\
                .withColumnRenamed('sum(qtde_venda)', 'qtd_venda_mensal')\
                .withColumnRenamed('sum(valor_venda)', 'valor_venda_mensal')\
                .withColumnRenamed('sum(valor_imposto)', 'valor_imposto_mensal')\
                .withColumnRenamed('sum(valor_custo)', 'valor_custo_mensal')

# Separando as variáveis categóricas
dataset_cat = dataset_v04.drop('qtde_venda', 'valor_venda', 'valor_imposto', 'valor_custo', 'data', 'cliente').distinct()

# Nova versão do dataset
dataset_v05 = dataset_cat.join(dataset_numeric_agg, on='produto_nome', how='left')

# Retirando Outliers
dataset_v06 = dataset_v05.where(col('qtd_venda_mensal') < 60).where(col('valor_venda_mensal') < 1500)\
                        .where(col('valor_imposto_mensal') < 200).where(col('valor_custo_mensal') < 600)

# Separando quantidade de meses desejada para modelagem
data_primeira = dataset_v06.select('data').distinct().orderBy('data', ascending = False).collect()[num_meses-1]
dataset_v07 = dataset_v06.where(col('data') >= data_primeira[0])

# Parametrizando coluna de datas
data_sequencia = dataset_v07.select('data').distinct().orderBy('data').withColumn('data_sequencia', 
                                                                                  (monotonically_increasing_id() + 1))
dataset_v08 = dataset_v07.join(data_sequencia, on='data', how='left').distinct()

# Para uma projeção de vendas de produtos, foram selecionadas variáveis contínuas para um modelo de regressão!
# Coluna de produtos
pk_col = 'produto_nome'
# Variáveis
x_cols = ['data_sequencia', 'valor_venda_mensal', 'valor_imposto_mensal', 'valor_custo_mensal']
# Variável resposta 'y'
target = 'qtd_venda_mensal'
df = dataset_v08.select('produto_nome', col('qtd_venda_mensal').cast('float'), col('data_sequencia').cast('float'), 
                        col('valor_venda_mensal').cast('float'), col('valor_imposto_mensal').cast('float'), 
                        col('valor_custo_mensal').cast('float')).distinct()
X = dataset_v08.select(col(x_cols[0]).cast('float'),
                       col(x_cols[1]).cast('float'),
                       col(x_cols[2]).cast('float'),
                       col(x_cols[3]).cast('float')).distinct()
y = dataset_v08.select(col(target).cast('float')).distinct()

# Modelagem
assembler = VectorAssembler(inputCols = x_cols, outputCol = 'features')
dadosFeatures = assembler.transform(df).select('qtd_venda_mensal', 'features')\
                                .withColumnRenamed('qtd_venda_mensal', 'label')
lr = LinearRegression()
lrModel = lr.fit(dadosFeatures_train)

# Preparando para predição da demanda dos produtos
data_sequencia = dataset_v05.select('data').distinct().orderBy('data').withColumn('data_sequencia', 
                                                                                  (monotonically_increasing_id() + 1))
dataset_predict = dataset_v05.join(data_sequencia, on='data', how='left').distinct()

# Dataset com as variáveis com valores do último mês estudado, porém, extrapolando a sequência de mês para os próximos

w = Window.partitionBy('produto_nome')
date_column = 'data_sequencia'

# 30 dias de extrapolação
df_7 = dataset_predict.withColumn('max_data', max(date_column).over(w))\
        .drop(date_column).distinct().withColumn('data_sequencia', lit(7)).drop('max_data')
assembler = VectorAssembler(inputCols = x_cols, outputCol = 'features')
dadosFeatures_7 = assembler.transform(df_7).withColumnRenamed('qtd_venda_mensal', 'label')
predict_30 = lrModel.transform(dadosFeatures_7)\
                    .withColumn('perc_aumento_demanda', ((col('prediction') - col('label'))/col('label'))*100)\
                        .select('produto_nome', 'prediction', 
                                'perc_aumento_demanda').withColumnRenamed('prediction', 'predicao_qtd_vendas')

# 60 dias de extrapolação
df_8 = dataset_predict.withColumn('max_data', max(date_column).over(w))\
        .drop(date_column).distinct().withColumn('data_sequencia', lit(8)).drop('max_data')
assembler = VectorAssembler(inputCols = x_cols, outputCol = 'features')
dadosFeatures_8 = assembler.transform(df_8).withColumnRenamed('qtd_venda_mensal', 'label')
predict_60 =lrModel.transform(dadosFeatures_8)\
                    .withColumn('perc_aumento_demanda', ((col('prediction') - col('label'))/col('label'))*100)\
                        .select('produto_nome', 'prediction', 
                                'perc_aumento_demanda').withColumnRenamed('prediction', 'predicao_qtd_vendas')

# 90 dias de extrapolação
df_9 = dataset_predict.withColumn('max_data', max(date_column).over(w))\
        .drop(date_column).distinct().withColumn('data_sequencia', lit(9)).drop('max_data')
assembler = VectorAssembler(inputCols = x_cols, outputCol = 'features')
dadosFeatures_9 = assembler.transform(df_9).withColumnRenamed('qtd_venda_mensal', 'label')
predict_90 = lrModel.transform(dadosFeatures_9)\
                    .withColumn('perc_aumento_demanda', ((col('prediction') - col('label'))/col('label'))*100)\
                        .select('produto_nome', 'prediction', 
                                'perc_aumento_demanda').withColumnRenamed('prediction', 'predicao_qtd_vendas')

# Persistindo na base os dados das predições realizadas!
predict_30.toPandas().to_csv(predict_30_save_local) # utilizando spark: .write.format('csv').mode('overwrite').saveAsTable()
predict_60.toPandas().to_csv(predict_60_save_local)
predict_90.toPandas().to_csv(predict_90_save_local)

# Créditos:
# Caio Cezar Veronezi Macedo