import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';

class OwnerManageWardensScreen extends StatefulWidget {
  final String ownerName;
  final String hostelName;
  final int? hostelId;

  const OwnerManageWardensScreen({
    super.key,
    required this.ownerName,
    required this.hostelName,
    this.hostelId,
  });

  @override
  State<OwnerManageWardensScreen> createState() =>
      _OwnerManageWardensScreenState();
}

class _OwnerManageWardensScreenState extends State<OwnerManageWardensScreen> {
  List<Map<String, dynamic>> _wardens = [];
  bool _loading = true;
  final _emailCtrl = TextEditingController();
  final _nameCtrl  = TextEditingController();
  bool _sending    = false;
  String? _inviteError;

  @override
  void initState() {
    super.initState();
    _loadWardens();
  }

  @override
  void dispose() {
    _emailCtrl.dispose();
    _nameCtrl.dispose();
    super.dispose();
  }

  // ── Load wardens from backend ──────────────────────────────────────────────
  Future<void> _loadWardens() async {
    try {
      final data = await Api().getOwnerWardens();
      if (!mounted) return;
      setState(() {
        _wardens = data.map<Map<String, dynamic>>((w) => {
          'id':     w['id'],
          'name':   w['name'] ?? 'Warden',
          'email':  w['email'] ?? '',
          'phone':  'Not provided',
          'status': 'Active',
          'joined': (w['created_at'] ?? '').toString().length >= 10
              ? (w['created_at'] as String).substring(0, 10)
              : 'Unknown',
          'floor':  'All Floors',
        }).toList();
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  // ── Add warden via backend ─────────────────────────────────────────────────
  Future<void> _sendInvite() async {
    final email = _emailCtrl.text.trim();
    final name  = _nameCtrl.text.trim();

    if (email.isEmpty || !email.contains('@')) {
      setState(() => _inviteError = 'Please enter a valid email address');
      return;
    }
    setState(() { _sending = true; _inviteError = null; });

    try {
      final result = await Api().addWarden(
        email: email,
        name: name.isNotEmpty ? name : null,
        hostelId: widget.hostelId,
      );

      if (!mounted) return;
      setState(() {
        _sending = false;
        _wardens.insert(0, {
          'id':     result['warden']?['id'],
          'name':   name.isNotEmpty ? name : 'Warden',
          'email':  email,
          'phone':  'Not provided',
          'status': 'Active',
          'joined': 'Just now',
          'floor':  'Not assigned',
          'temp_password': result['temp_password'],
        });
        _emailCtrl.clear();
        _nameCtrl.clear();
      });

      Navigator.pop(context); // close sheet

      // Show credentials to owner
      if (result['temp_password'] != null) {
        _showCredentials(
          name: name.isNotEmpty ? name : 'Warden',
          email: email,
          password: result['temp_password'] as String,
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Warden linked successfully ✅', style: GoogleFonts.dmSans()),
          backgroundColor: AppTheme.teal,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          margin: const EdgeInsets.all(16)));
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _sending = false;
        _inviteError = e.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  // ── Show credentials dialog ────────────────────────────────────────────────
  void _showCredentials({
    required String name,
    required String email,
    required String password,
  }) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        backgroundColor: AppTheme.card,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(children: [
          Container(width: 38, height: 38,
            decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.1), shape: BoxShape.circle),
            child: const Icon(Icons.key_rounded, color: AppTheme.teal, size: 20)),
          const SizedBox(width: 10),
          Text('Warden Credentials', style: GoogleFonts.dmSans(
            fontSize: 16, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
        ]),
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          Text('Share these with $name', style: GoogleFonts.dmSans(
            fontSize: 13, color: AppTheme.textMuted)),
          const SizedBox(height: 14),
          _credRow('Name', name),
          const SizedBox(height: 8),
          _credRow('Email', email),
          const SizedBox(height: 8),
          _credRow('Password', password),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(color: Colors.amber.shade50,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.amber.shade200, width: 0.5)),
            child: Row(children: [
              Icon(Icons.warning_amber_rounded, color: Colors.amber.shade700, size: 15),
              const SizedBox(width: 8),
              Expanded(child: Text(
                'Share these credentials securely. Warden should change password after first login.',
                style: GoogleFonts.dmSans(fontSize: 11, color: Colors.amber.shade800))),
            ]),
          ),
        ]),
        actions: [
          TextButton(
            onPressed: () {
              Clipboard.setData(ClipboardData(
                text: 'StayBuddy Warden Login\nEmail: $email\nPassword: $password'));
              ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                content: Text('Credentials copied! ✅', style: GoogleFonts.dmSans()),
                backgroundColor: AppTheme.teal,
                behavior: SnackBarBehavior.floating,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                margin: const EdgeInsets.all(16)));
            },
            child: Text('Copy', style: GoogleFonts.dmSans(
              color: AppTheme.teal, fontWeight: FontWeight.w600)),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
              elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10))),
            child: Text('Done', style: GoogleFonts.dmSans(fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }

  Widget _credRow(String label, String value) => Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(color: AppTheme.bg, borderRadius: BorderRadius.circular(10),
      border: Border.all(color: AppTheme.border, width: 0.5)),
    child: Row(children: [
      SizedBox(width: 70, child: Text('$label:', style: GoogleFonts.dmSans(
        fontSize: 12, color: AppTheme.textMuted))),
      Expanded(child: Text(value, style: GoogleFonts.dmSans(
        fontSize: 13, fontWeight: FontWeight.w700, color: AppTheme.textPrimary))),
      GestureDetector(
        onTap: () => Clipboard.setData(ClipboardData(text: value)),
        child: const Icon(Icons.copy_rounded, size: 14, color: AppTheme.teal)),
    ]),
  );

  // ── Add warden bottom sheet ────────────────────────────────────────────────
  void _showAddWardenSheet() {
    showModalBottomSheet(
      context: context, isScrollControlled: true, backgroundColor: Colors.transparent,
      builder: (_) => Padding(
        padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
        child: Container(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
          decoration: const BoxDecoration(color: AppTheme.bg,
            borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Center(child: Container(width: 40, height: 4,
              margin: const EdgeInsets.only(bottom: 18),
              decoration: BoxDecoration(color: AppTheme.border, borderRadius: BorderRadius.circular(2)))),

            Row(children: [
              Container(width: 38, height: 38,
                decoration: BoxDecoration(
                  color: const Color(0xFF185FA5).withOpacity(0.1), shape: BoxShape.circle),
                child: const Icon(Icons.person_add_rounded, color: Color(0xFF185FA5), size: 18)),
              const SizedBox(width: 10),
              Text('Add New Warden', style: GoogleFonts.playfairDisplay(
                fontSize: 18, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
            ]),
            const SizedBox(height: 6),
            Text('A warden account will be created and credentials shown to you.',
              style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted, height: 1.4)),
            const SizedBox(height: 18),

            _field(_nameCtrl, 'Warden Name (optional)', Icons.person_outline_rounded),
            const SizedBox(height: 10),
            _field(_emailCtrl, 'Warden Email Address *',
              Icons.email_outlined, type: TextInputType.emailAddress),

            if (_inviteError != null) ...[
              const SizedBox(height: 8),
              Container(padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.red.shade200, width: 0.5)),
                child: Row(children: [
                  Icon(Icons.error_outline_rounded, color: Colors.red.shade600, size: 14),
                  const SizedBox(width: 8),
                  Expanded(child: Text(_inviteError!, style: GoogleFonts.dmSans(
                    fontSize: 12, color: Colors.red.shade700))),
                ])),
            ],
            const SizedBox(height: 14),

            // Flow info
            Container(padding: const EdgeInsets.all(13),
              decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.05),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppTheme.teal.withOpacity(0.2))),
              child: Column(children: [
                _flowStep('1', 'Enter warden name and email'),
                _flowStep('2', 'System creates warden account'),
                _flowStep('3', 'Credentials shown to you'),
                _flowStep('4', 'Share credentials with warden'),
                _flowStep('5', 'Warden logs in → manages hostel'),
              ])),
            const SizedBox(height: 18),

            SizedBox(width: double.infinity, height: 50,
              child: ElevatedButton(
                onPressed: _sending ? null : _sendInvite,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF185FA5), foregroundColor: Colors.white,
                  elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
                child: _sending
                    ? const SizedBox(width: 20, height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                    : Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                        const Icon(Icons.person_add_rounded, size: 16),
                        const SizedBox(width: 8),
                        Text('Create Warden Account', style: GoogleFonts.dmSans(
                          fontSize: 15, fontWeight: FontWeight.w600)),
                      ]),
              )),
          ]),
        ),
      ),
    );
  }

  Widget _flowStep(String num, String text) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 3),
    child: Row(children: [
      Container(width: 20, height: 20,
        decoration: const BoxDecoration(color: AppTheme.teal, shape: BoxShape.circle),
        child: Center(child: Text(num, style: GoogleFonts.dmSans(
          fontSize: 10, fontWeight: FontWeight.w700, color: Colors.white)))),
      const SizedBox(width: 10),
      Text(text, style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.teal)),
    ]),
  );

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        appBar: AppBar(
          title: const Text('Manage Wardens'),
          leading: IconButton(
            icon: const Icon(Icons.arrow_back_rounded),
            onPressed: () => Navigator.pop(context)),
          actions: [
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: ElevatedButton.icon(
                onPressed: _showAddWardenSheet,
                icon: const Icon(Icons.person_add_rounded, size: 16),
                label: Text('Add', style: GoogleFonts.dmSans(
                  fontSize: 13, fontWeight: FontWeight.w600)),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white.withOpacity(0.2),
                  foregroundColor: Colors.white, elevation: 0,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20))),
              ),
            ),
          ],
        ),
        body: _loading
            ? const Center(child: CircularProgressIndicator())
            : Column(children: [
                // Info banner
                Container(
                  color: AppTheme.tealDeep,
                  padding: const EdgeInsets.fromLTRB(16, 10, 16, 14),
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.08),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white.withOpacity(0.15))),
                    child: Row(children: [
                      const Icon(Icons.info_outline_rounded, color: AppTheme.tealMint, size: 16),
                      const SizedBox(width: 10),
                      Expanded(child: Text(
                        'Add wardens by email. System creates account and shows credentials to share.',
                        style: GoogleFonts.dmSans(fontSize: 12, color: Colors.white70, height: 1.4))),
                    ]),
                  ),
                ),

                // Wardens list
                Expanded(
                  child: _wardens.isEmpty
                      ? _buildEmpty()
                      : RefreshIndicator(
                          onRefresh: _loadWardens,
                          child: ListView.builder(
                            padding: const EdgeInsets.fromLTRB(16, 14, 16, 24),
                            itemCount: _wardens.length,
                            itemBuilder: (_, i) => _wardenCard(_wardens[i]),
                          ),
                        ),
                ),
              ]),
        floatingActionButton: FloatingActionButton.extended(
          onPressed: _showAddWardenSheet,
          backgroundColor: AppTheme.teal,
          icon: const Icon(Icons.person_add_rounded, color: Colors.white),
          label: Text('Add Warden', style: GoogleFonts.dmSans(
            color: Colors.white, fontWeight: FontWeight.w600)),
        ),
      ),
    );
  }

  Widget _wardenCard(Map<String, dynamic> w) {
    final status = w['status'] as String;
    final isActive = status == 'Active';
    final statusColor = isActive ? AppTheme.green : Colors.orange.shade700;
    final statusBg = isActive ? AppTheme.greenLight : Colors.orange.shade50;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.card, borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.border, width: 0.5),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04),
          blurRadius: 8, offset: const Offset(0, 2))]),
      child: Column(children: [
        Row(children: [
          Container(width: 48, height: 48,
            decoration: BoxDecoration(
              gradient: LinearGradient(colors: [
                const Color(0xFF185FA5).withOpacity(0.15),
                const Color(0xFF185FA5).withOpacity(0.08)]),
              shape: BoxShape.circle),
            child: const Icon(Icons.badge_rounded, color: Color(0xFF185FA5), size: 22)),
          const SizedBox(width: 12),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(w['name'] as String, style: GoogleFonts.dmSans(
              fontSize: 14, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
            Text(w['email'] as String, style: GoogleFonts.dmSans(
              fontSize: 11, color: AppTheme.textMuted)),
          ])),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(color: statusBg, borderRadius: BorderRadius.circular(8)),
            child: Text(status, style: GoogleFonts.dmSans(
              fontSize: 11, fontWeight: FontWeight.w700, color: statusColor))),
        ]),
        const SizedBox(height: 10),
        Divider(height: 0.5, color: AppTheme.border, thickness: 0.5),
        const SizedBox(height: 10),
        Row(children: [
          _infoChip(Icons.calendar_today_rounded, w['joined'] as String),
          const SizedBox(width: 12),
          _infoChip(Icons.layers_rounded, w['floor'] as String),
          const Spacer(),
          // Show credentials button if temp password exists
          if (w['temp_password'] != null)
            GestureDetector(
              onTap: () => _showCredentials(
                name: w['name'] as String,
                email: w['email'] as String,
                password: w['temp_password'] as String),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8)),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  const Icon(Icons.key_rounded, size: 12, color: AppTheme.teal),
                  const SizedBox(width: 4),
                  Text('View Credentials', style: GoogleFonts.dmSans(
                    fontSize: 10, fontWeight: FontWeight.w600, color: AppTheme.teal)),
                ]),
              ),
            ),
        ]),
      ]),
    );
  }

  Widget _infoChip(IconData icon, String text) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Icon(icon, size: 11, color: AppTheme.textMuted),
      const SizedBox(width: 4),
      Text(text, style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted)),
    ],
  );

  Widget _buildEmpty() => Center(
    child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      Icon(Icons.badge_outlined, size: 64, color: AppTheme.textMuted.withOpacity(0.3)),
      const SizedBox(height: 14),
      Text('No wardens yet', style: GoogleFonts.dmSans(fontSize: 16, color: AppTheme.textMuted)),
      const SizedBox(height: 6),
      Text('Add a warden to manage your hostel',
        style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textMuted)),
      const SizedBox(height: 20),
      ElevatedButton.icon(
        onPressed: _showAddWardenSheet,
        icon: const Icon(Icons.person_add_rounded, size: 16),
        label: Text('Add First Warden', style: GoogleFonts.dmSans(fontWeight: FontWeight.w600)),
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
          elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
      ),
    ]),
  );

  Widget _field(TextEditingController ctrl, String hint, IconData icon,
      {TextInputType? type}) =>
      Container(
        decoration: BoxDecoration(color: AppTheme.card, borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppTheme.border, width: 0.5)),
        child: TextField(
          controller: ctrl, keyboardType: type,
          style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
          decoration: InputDecoration(
            hintText: hint,
            hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
            prefixIcon: Icon(icon, size: 18, color: AppTheme.textMuted),
            border: InputBorder.none,
            contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14)),
        ),
      );
}
