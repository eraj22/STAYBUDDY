import 'package:flutter/material.dart';

import '../../api.dart';

class ParentEmergencyScreen extends StatefulWidget {
  const ParentEmergencyScreen({super.key});

  @override
  State<ParentEmergencyScreen> createState() => _ParentEmergencyScreenState();
}

class _ParentEmergencyScreenState extends State<ParentEmergencyScreen> {
  bool _loading = true;
  String? _error;
  String? _childName;
  List<dynamic> _incidents = const [];

  @override
  void initState() { super.initState(); _load(); }

  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final children = await Api().getParentChildren();
      if (children.isEmpty) throw Exception('No linked student was found.');
      final child = Map<String, dynamic>.from(children.first as Map);
      final id = int.tryParse(child['id']?.toString() ?? '');
      if (id == null) throw Exception('The linked student record is invalid.');
      final incidents = await Api().getParentChildEmergencies(id);
      if (mounted) setState(() { _childName = child['name']?.toString() ?? 'your child'; _incidents = incidents; });
    } catch (error) {
      if (mounted) setState(() => _error = error.toString().replaceFirst('Exception: ', ''));
    } finally { if (mounted) setState(() => _loading = false); }
  }

  Color _statusColor(String status) => switch (status) {
    'resolved' || 'cancelled' => Colors.green,
    'acknowledged' => Colors.orange,
    _ => Colors.red,
  };

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Emergency Updates')),
      body: RefreshIndicator(
        onRefresh: _load,
        child: _loading ? const Center(child: CircularProgressIndicator())
          : _error != null ? ListView(children: [Padding(padding: const EdgeInsets.all(24), child: Column(children: [Text(_error!, textAlign: TextAlign.center), const SizedBox(height: 12), FilledButton(onPressed: _load, child: const Text('Retry'))]))])
          : _incidents.isEmpty ? ListView(children: [const SizedBox(height: 110), const Icon(Icons.health_and_safety_outlined, size: 64), const SizedBox(height: 16), Center(child: Text('No emergency incidents are recorded for ${_childName ?? 'your child'}.'))])
          : ListView.separated(
              padding: const EdgeInsets.all(16), itemCount: _incidents.length, separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (_, index) { final item = Map<String, dynamic>.from(_incidents[index] as Map); final status = item['status']?.toString() ?? 'active'; return Card(child: ListTile(
                isThreeLine: true, leading: Icon(Icons.emergency, color: _statusColor(status)),
                title: Text('${item['alert_type'] ?? 'Emergency'} • ${status.toUpperCase()}'),
                subtitle: Text('${item['description'] ?? 'No description supplied'}\n${item['created_at'] ?? ''}'),
                trailing: Chip(label: Text(status), backgroundColor: _statusColor(status).withOpacity(.12)),
              )); },
            ),
      ),
    );
  }
}
