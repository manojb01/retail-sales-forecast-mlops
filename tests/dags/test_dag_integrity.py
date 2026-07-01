import pytest
from airflow.models import DagBag
import os


class TestDagIntegrity:

    @pytest.fixture
    def dagbag(self):
        """Create a DagBag for testing"""
        dag_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dags')
        return DagBag(dag_folder=dag_folder, include_examples=False)

    def test_dag_bag_import(self, dagbag):
        """Test that DagBag imports successfully with no errors"""
        assert dagbag.import_errors == {}, f"DAG import errors: {dagbag.import_errors}"

    def test_dag_loaded(self, dagbag):
        """Test that expected DAGs are loaded"""
        # Only test DAGs that actually exist
        expected_dags = [
            'sales_forecast_training',
        ]

        for dag_id in expected_dags:
            assert dag_id in dagbag.dags, f"DAG {dag_id} not found in DagBag"

    def test_dag_structure(self, dagbag):
        """Test DAG structure and dependencies"""
        # Test training DAG
        training_dag = dagbag.get_dag('sales_forecast_training')
        assert training_dag is not None

        # Check task count (8 tasks in the training DAG)
        assert len(training_dag.tasks) >= 7, "Training DAG should have at least 7 tasks"

        # Check critical tasks exist (TaskFlow API adds _task suffix)
        task_ids = [task.task_id for task in training_dag.tasks]
        expected_tasks = [
            'extract_data_task',
            'validate_data_task',
            'train_models_task',
            'evaluate_models_task',
            'register_best_model_task',
            'transition_to_production_task',
            'generate_performance_report_task',
            'cleanup'
        ]
        for task_id in expected_tasks:
            assert task_id in task_ids, f"Task {task_id} not found in training DAG"

    def test_dag_task_dependencies(self, dagbag):
        """Test that task dependencies are correctly set"""
        training_dag = dagbag.get_dag('sales_forecast_training')

        # Get tasks
        extract_task = training_dag.get_task('extract_data_task')
        validate_task = training_dag.get_task('validate_data_task')
        train_task = training_dag.get_task('train_models_task')
        cleanup_task = training_dag.get_task('cleanup')
        report_task = training_dag.get_task('generate_performance_report_task')

        # Check extract -> validate dependency
        assert validate_task in extract_task.downstream_list, \
            "validate_data_task should be downstream of extract_data_task"

        # Check report -> cleanup dependency
        assert cleanup_task in report_task.downstream_list, \
            "cleanup should be downstream of generate_performance_report_task"

    def test_dag_schedule_interval(self, dagbag):
        """Test that DAGs have appropriate schedule intervals"""
        training_dag = dagbag.get_dag('sales_forecast_training')
        # Airflow 3.0 uses 'schedule' instead of 'schedule_interval'
        schedule = getattr(training_dag, 'schedule', None) or getattr(training_dag, 'schedule_interval', None)
        assert schedule == '@weekly', \
            f"Training DAG has schedule {schedule}, expected @weekly"

    def test_dag_tags(self, dagbag):
        """Test that DAGs have appropriate tags"""
        training_dag = dagbag.get_dag('sales_forecast_training')
        assert 'ml' in training_dag.tags, "Training DAG should have 'ml' tag"
        assert 'training' in training_dag.tags, "Training DAG should have 'training' tag"

    def test_no_import_errors(self, dagbag):
        """Test that there are no import errors in any DAG"""
        assert len(dagbag.import_errors) == 0, \
            f"Found import errors: {dagbag.import_errors}"

    def test_dag_retries(self, dagbag):
        """Test that all DAGs have retry configuration"""
        for dag_id, dag in dagbag.dags.items():
            assert dag.default_args.get('retries', 0) >= 1, \
                f"DAG {dag_id} should have at least 1 retry configured"

    def test_dag_emails(self, dagbag):
        """Test that all DAGs have email configuration"""
        for dag_id, dag in dagbag.dags.items():
            assert 'email' in dag.default_args, \
                f"DAG {dag_id} should have email configuration"
            assert dag.default_args.get('email_on_failure', False) is True, \
                f"DAG {dag_id} should have email_on_failure enabled"
