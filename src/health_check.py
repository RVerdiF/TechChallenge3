from flask import Flask, request
import logging

# Desativa o logger padrão do Flask para evitar saídas duplicadas no console
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    """
    Endpoint para health checks.
    Responde com 200 OK para requisições GET e HEAD.
    """
    # Para requisições HEAD, o Flask cuida de retornar apenas os cabeçalhos.
    # Para GET, ele retornará o corpo 'OK'.
    return 'OK', 200

def run_health_check_server():
    """
    Inicia o servidor Flask na porta 8502.
    Isso deve ser executado em uma thread separada da aplicação Streamlit.
    """
    # Executando em uma porta diferente da padrão do Streamlit (8501)
    app.run(host='0.0.0.0', port=8502)

if __name__ == '__main__':
    # Permite executar o servidor de health check diretamente para testes
    print("Iniciando servidor de health check Flask em http://0.0.0.0:8502/health")
    run_health_check_server()
