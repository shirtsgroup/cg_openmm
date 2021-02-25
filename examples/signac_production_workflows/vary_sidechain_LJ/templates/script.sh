{% extends base_script %}
{% block body %}
module load cuda/10.1
{{ super() }}
{% endblock %}