import 'package:flutter/material.dart';

import '../../api.dart';

class ParentTransportScreen extends StatefulWidget {
  const ParentTransportScreen({super.key});

  @override
  State<ParentTransportScreen> createState() => _ParentTransportScreenState();
}

class _ParentTransportScreenState extends State<ParentTransportScreen> {
  bool _loading = true;
  String? _error;
  String? _childName;
  List<dynamic> _assignments = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final children = await Api().getParentChildren();
      if (children.isEmpty) {
        if (mounted) setState(() { _assignments = const []; _childName = null; });
        return;
      }
      final child = Map<String, dynamic>.from(children.first as Map);
      final id = int.tryParse(child['id']?.toString() ?? '');
      if (id == null) throw Exception('The linked student record is invalid.');
      final assignments = await Api().getParentChildTransport(id);
      if (mounted) setState(() {
        _assignments = assignments;
        _childName = child['name']?.toString() ?? 'your child';
      });
    } catch (error) {
      if (mounted) setState(() => _error = error.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String _format(dynamic value) {
    if (value == null || value.toString().isEmpty) return 'Not provided';
    return value.toString();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Transport Details')),
      body: RefreshIndicator(
        onRefresh: _load,
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : _error != null
                ? ListView(children: [Padding(padding: const EdgeInsets.all(24), child: Text(_error!)), Center(child: FilledButton(onPressed: _load, child: const Text('Retry')))])
                : _assignments.isEmpty
                    ? ListView(children: [const SizedBox(height: 120), const Icon(Icons.directions_bus_outlined, size: 64), const SizedBox(height: 16), Center(child: Text(_childName == null ? 'No linked student was found.' : 'No active transport service is assigned to $_childName.'))])
                    : ListView.separated(
                        padding: const EdgeInsets.all(16),
                        itemCount: _assignments.length,
                        separatorBuilder: (_, __) => const SizedBox(height: 12),
                        itemBuilder: (_, index) => _assignmentCard(Map<String, dynamic>.from(_assignments[index] as Map)),
                      ),
      ),
    );
  }

  Widget _assignmentCard(Map<String, dynamic> item) {
    final events = List<dynamic>.from(item['recent_events'] as List? ?? const []);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [const Icon(Icons.directions_bus), const SizedBox(width: 8), Expanded(child: Text(_format(item['service_name']), style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)))]),
          const SizedBox(height: 12),
          Text('Mode: ${_format(item['mode'])}'),
          Text('Vehicle: ${_format(item['vehicle_number'])}'),
          Text('Driver: ${_format(item['driver_name'])}  ${_format(item['driver_phone'])}'),
          const Divider(height: 24),
          Text('Pickup: ${_format(item['pickup_stop'])}  ${_format(item['pickup_time'])}'),
          Text('Drop-off: ${_format(item['dropoff_stop'])}  ${_format(item['dropoff_time'])}'),
          if (events.isNotEmpty) ...[
            const SizedBox(height: 16), const Text('Recent updates', style: TextStyle(fontWeight: FontWeight.bold)),
            ...events.take(3).map((event) { final e = Map<String, dynamic>.from(event as Map); return ListTile(contentPadding: EdgeInsets.zero, dense: true, leading: const Icon(Icons.history), title: Text(_format(e['event_type'])), subtitle: Text(_format(e['occurred_at'])), trailing: e['note'] == null ? null : Text(_format(e['note']), overflow: TextOverflow.ellipsis)); }),
          ],
        ]),
      ),
    );
  }
}
