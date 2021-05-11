{% extends base_script %}
{% block body %}
module load cuda/9.2
{{ super() }}
{% endblock %}